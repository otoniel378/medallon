"""
forecasting.py — Pronosticos de demanda mensual.
Modelos disponibles:
  - ETS  : Holt-Winters Triple Exponential Smoothing (primario — más estable)
  - SARIMA: SARIMA/ARIMA via statsmodels (secundario)
  - XGB  : XGBoost con lag features (ML no paramétrico)
  - NAIVE: Seasonal Naive — último año disponible como pronóstico (fallback)
"""

import pandas as pd
import numpy as np
import warnings
from dataclasses import dataclass
from config import MIN_PERIODS_FORECAST


# ─────────────────────────────────────────────
# Contenedor de resultados
# ─────────────────────────────────────────────
@dataclass
class ForecastResult:
    modelo: str
    historico: pd.DataFrame    # ds, y
    forecast: pd.DataFrame     # ds, yhat, yhat_lower, yhat_upper
    metricas: dict             # MAE, MAPE, RMSE
    backtest: pd.DataFrame     # ds, y_real, y_pred
    error_msg: str | None = None


# ─────────────────────────────────────────────
# Métricas de error
# ─────────────────────────────────────────────
def _metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    y_real = np.array(y_real, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae  = float(np.mean(np.abs(y_real - y_pred)))
    rmse = float(np.sqrt(np.mean((y_real - y_pred) ** 2)))
    mask = y_real > 0
    mape = float(np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100) if mask.sum() > 0 else float("nan")
    return {"MAE": round(mae, 2), "MAPE (%)": round(mape, 1), "RMSE": round(rmse, 2)}


def _preparar_serie(df, col_periodo="PERIODO", col_val="PESO_TON") -> pd.DataFrame:
    s = df[[col_periodo, col_val]].copy()
    s.columns = ["ds", "y"]
    s["ds"] = pd.to_datetime(s["ds"])
    s["y"]  = pd.to_numeric(s["y"], errors="coerce").fillna(0).clip(lower=0)
    s = s.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    # Eliminar el mes actual si está incompleto (mes corriente aún no ha terminado)
    if not s.empty:
        today = pd.Timestamp.now()
        current_month_start = today.to_period("M").to_timestamp()
        if s["ds"].max() >= current_month_start:
            s = s[s["ds"] < current_month_start].reset_index(drop=True)
    return s


def _build_hist_fc(serie, fc_values, lower, upper) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (forecast_df, hist_part) compatibles con la página."""
    last_date = serie["ds"].max()
    h = len(fc_values)
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1), periods=h, freq="MS"
    )
    hist_part = serie[["ds", "y"]].rename(columns={"y": "yhat"}).copy()
    hist_part["yhat_lower"] = np.nan
    hist_part["yhat_upper"] = np.nan
    fc_part = pd.DataFrame({
        "ds": future_dates,
        "yhat":       np.clip(np.asarray(fc_values).ravel(), 0, None),
        "yhat_lower": np.clip(np.asarray(lower).ravel(), 0, None),
        "yhat_upper": np.asarray(upper).ravel(),
    })
    return pd.concat([hist_part, fc_part], ignore_index=True)


# ═══════════════════════════════════════════════════════════════
# MODELO 1 — Holt-Winters ETS  (primario)
# ═══════════════════════════════════════════════════════════════
def _forecast_ets(serie: pd.DataFrame, horizonte: int) -> ForecastResult:
    """
    Triple Exponential Smoothing (Holt-Winters).
    Captura nivel + tendencia + estacionalidad multiplicativa/aditiva.
    Más estable que SARIMA para series cortas o con alta variabilidad.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = serie["y"].values
    n = len(y)
    n_test = min(6, n // 5)
    y_train = y[:-n_test] if n_test > 0 else y
    y_test  = y[-n_test:]  if n_test > 0 else np.array([])

    seasonal = "add" if n >= 24 else None
    sp       = 12   if seasonal else None
    trend    = "add"

    def _fit(arr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return ExponentialSmoothing(
                    arr, trend=trend, seasonal=seasonal,
                    seasonal_periods=sp, initialization_method="estimated",
                    use_boxcox=False,
                ).fit(optimized=True, remove_bias=False)
            except Exception:
                return ExponentialSmoothing(arr, trend="add").fit(optimized=True)

    try:
        backtest_df, metricas = pd.DataFrame(), {}
        if n_test > 0:
            m_bt  = _fit(y_train)
            p_bt  = np.clip(m_bt.forecast(n_test), 0, None)
            backtest_df = pd.DataFrame({
                "ds": serie["ds"].iloc[-n_test:].values,
                "y_real": y_test, "y_pred": p_bt,
            })
            metricas = _metricas(y_test, p_bt)

        m_full   = _fit(y)
        fc_vals  = m_full.forecast(horizonte)
        residuals = np.asarray(m_full.resid)
        sigma    = float(np.std(residuals)) * np.sqrt(np.arange(1, horizonte + 1))
        z90      = 1.645
        lower    = fc_vals - z90 * sigma
        upper    = fc_vals + z90 * sigma

        forecast_df = _build_hist_fc(serie, fc_vals, lower, upper)
        return ForecastResult(
            modelo="Holt-Winters (ETS)",
            historico=serie, forecast=forecast_df,
            metricas=metricas, backtest=backtest_df,
        )
    except Exception as e:
        return ForecastResult(
            modelo="ETS", historico=serie,
            forecast=pd.DataFrame(), metricas={}, backtest=pd.DataFrame(),
            error_msg=str(e),
        )


# ═══════════════════════════════════════════════════════════════
# MODELO 2 — SARIMA  (estadístico clásico)
# ═══════════════════════════════════════════════════════════════
def _forecast_sarima(serie: pd.DataFrame, horizonte: int) -> ForecastResult:
    """
    SARIMA entrenado desde SARIMA_FECHA_INICIO en adelante.
    Captura el regimen actual de demanda (descendente y estabilizado desde 2022).
    Diferenciacion conservadora para evitar sobre-diferenciacion.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Recortar al regimen actual — solo datos desde 2022 en adelante
    serie = serie[serie["ds"] >= SARIMA_FECHA_INICIO].reset_index(drop=True)

    y = serie["y"].values
    n = len(y)
    n_test = min(4, n // 6)
    y_train = y[:-n_test] if n_test > 0 else y
    y_test  = y[-n_test:]  if n_test > 0 else np.array([])

    # Orden conservador: solo diferencia ordinaria, sin diferencia estacional
    # para evitar sobre-diferenciación que causa MAPE explosivo
    if n >= 36:
        s_order = (1, 0, 1, 12)   # estacional sin doble diferenciación
        nombre  = "SARIMA(1,1,1)(1,0,1)₁₂"
    elif n >= 24:
        s_order = (0, 0, 1, 12)
        nombre  = "SARIMA(1,1,1)(0,0,1)₁₂"
    else:
        s_order = (0, 0, 0, 0)
        nombre  = "ARIMA(1,1,1)"

    def _fit(arr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return SARIMAX(
                    arr, order=(1, 1, 1), seasonal_order=s_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False, maxiter=300)
            except Exception:
                from statsmodels.tsa.arima.model import ARIMA as _A
                return _A(arr, order=(1, 1, 1)).fit()

    try:
        backtest_df, metricas = pd.DataFrame(), {}
        if n_test > 0:
            m_bt = _fit(y_train)
            p_bt = np.clip(np.asarray(m_bt.forecast(n_test)).ravel(), 0, None)
            backtest_df = pd.DataFrame({
                "ds": serie["ds"].iloc[-n_test:].values,
                "y_real": y_test, "y_pred": p_bt,
            })
            metricas = _metricas(y_test, p_bt)

        m_full  = _fit(y)
        fc_obj  = m_full.get_forecast(steps=horizonte)
        try:
            fc_frame = fc_obj.summary_frame(alpha=0.10)
            fc_vals  = np.clip(fc_frame["mean"].values, 0, None)
            lower    = np.clip(fc_frame["mean_ci_lower"].values, 0, None)
            upper    = fc_frame["mean_ci_upper"].values
        except Exception:
            fc_vals = np.clip(np.asarray(fc_obj.predicted_mean).ravel(), 0, None)
            lower   = fc_vals * 0.80
            upper   = fc_vals * 1.20

        forecast_df = _build_hist_fc(serie, fc_vals, lower, upper)
        return ForecastResult(
            modelo=nombre,
            historico=serie, forecast=forecast_df,
            metricas=metricas, backtest=backtest_df,
        )
    except Exception as e:
        return ForecastResult(
            modelo="SARIMA", historico=serie,
            forecast=pd.DataFrame(), metricas={}, backtest=pd.DataFrame(),
            error_msg=str(e),
        )


# ═══════════════════════════════════════════════════════════════
# MODELO 3 — XGBoost con lag features  (ML)
# ═══════════════════════════════════════════════════════════════
def _forecast_xgboost(serie: pd.DataFrame, horizonte: int) -> ForecastResult:
    """
    XGBoost con ingeniería de features temporales.
    Captura patrones no lineales sin supuestos estadísticos.

    Features usados:
      - Rezagos 1..12 (o hasta n/3)
      - Media móvil 3, 6, 12 meses
      - Mes del año, trimestre (estacionalidad implícita)
      - Tendencia lineal
    """
    try:
        import xgboost as xgb
    except ImportError:
        return ForecastResult(
            modelo="XGBoost", historico=serie,
            forecast=pd.DataFrame(), metricas={}, backtest=pd.DataFrame(),
            error_msg="xgboost no instalado. Ejecuta: pip install xgboost",
        )

    y  = serie["y"].values.astype(float)
    ds = pd.to_datetime(serie["ds"].values)
    n  = len(y)
    MAX_LAG = min(12, n // 3)

    if MAX_LAG < 2:
        return ForecastResult(
            modelo="XGBoost", historico=serie,
            forecast=pd.DataFrame(), metricas={}, backtest=pd.DataFrame(),
            error_msg=f"Serie demasiado corta para XGBoost (n={n}).",
        )

    # ── Construcción de features ──────────────────────────────
    def make_X(y_arr, ds_arr):
        rows = []
        for i in range(MAX_LAG, len(y_arr)):
            row = {}
            for lag in range(1, MAX_LAG + 1):
                row[f"lag_{lag}"] = y_arr[i - lag]
            row["roll_mean_3"]  = float(np.mean(y_arr[max(0, i-3):i]))
            row["roll_mean_6"]  = float(np.mean(y_arr[max(0, i-6):i]))
            row["roll_mean_12"] = float(np.mean(y_arr[max(0, i-12):i]))
            row["roll_std_6"]   = float(np.std(y_arr[max(0, i-6):i])  + 1e-9)
            dt = pd.Timestamp(ds_arr[i])
            row["month"]    = dt.month
            row["quarter"]  = dt.quarter
            row["trend_t"]  = i  # tendencia lineal
            rows.append(row)
        return pd.DataFrame(rows)

    X_all = make_X(y, ds)
    y_all = y[MAX_LAG:]

    # ── Backtesting ────────────────────────────────────────────
    n_test = min(6, len(X_all) // 4)
    backtest_df, metricas = pd.DataFrame(), {}
    model = None

    try:
        if n_test > 0:
            X_tr, X_te = X_all.iloc[:-n_test], X_all.iloc[-n_test:]
            y_tr, y_te = y_all[:-n_test],       y_all[-n_test:]
            m_bt = xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="reg:squarederror", random_state=42, verbosity=0,
            )
            m_bt.fit(X_tr, y_tr,
                     eval_set=[(X_te, y_te)],
                     verbose=False)
            p_bt = np.clip(m_bt.predict(X_te), 0, None)
            backtest_df = pd.DataFrame({
                "ds": serie["ds"].iloc[-(n_test):].values,
                "y_real": y_te, "y_pred": p_bt,
            })
            metricas = _metricas(y_te, p_bt)

        # Modelo final con todos los datos
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42, verbosity=0,
        )
        model.fit(X_all, y_all, verbose=False)

        # ── Pronóstico iterativo ───────────────────────────────
        y_ext  = list(y.copy())     # historia extendida
        ds_ext = list(ds.copy())
        fc_vals, fc_lower, fc_upper = [], [], []

        last_trend = len(y_ext) - 1

        for h in range(horizonte):
            next_ds = ds_ext[-1] + pd.offsets.MonthBegin(1)
            next_t  = last_trend + h + 1
            y_arr_ext = np.array(y_ext)

            row = {}
            for lag in range(1, MAX_LAG + 1):
                idx = len(y_ext) - lag
                row[f"lag_{lag}"] = float(y_arr_ext[idx]) if idx >= 0 else 0.0
            row["roll_mean_3"]  = float(np.mean(y_arr_ext[-3:]))
            row["roll_mean_6"]  = float(np.mean(y_arr_ext[-6:]))
            row["roll_mean_12"] = float(np.mean(y_arr_ext[-12:]) if len(y_arr_ext) >= 12 else np.mean(y_arr_ext))
            row["roll_std_6"]   = float(np.std(y_arr_ext[-6:]) + 1e-9)
            row["month"]   = next_ds.month
            row["quarter"] = next_ds.quarter
            row["trend_t"] = next_t

            X_pred = pd.DataFrame([row])
            yhat   = float(np.clip(model.predict(X_pred)[0], 0, None))

            # Intervalo de confianza basado en residuos del modelo
            sigma_model = float(np.std(model.predict(X_all) - y_all) + 1e-9)
            margin = 1.645 * sigma_model * np.sqrt(h + 1)

            fc_vals.append(yhat)
            fc_lower.append(max(0.0, yhat - margin))
            fc_upper.append(yhat + margin)
            y_ext.append(yhat)
            ds_ext.append(next_ds)

        forecast_df = _build_hist_fc(serie, fc_vals, fc_lower, fc_upper)
        return ForecastResult(
            modelo="XGBoost (Lag Features)",
            historico=serie, forecast=forecast_df,
            metricas=metricas, backtest=backtest_df,
        )

    except Exception as e:
        return ForecastResult(
            modelo="XGBoost", historico=serie,
            forecast=pd.DataFrame(), metricas={}, backtest=pd.DataFrame(),
            error_msg=str(e),
        )


# ═══════════════════════════════════════════════════════════════
# MODELO 4 — Seasonal Naive  (baseline / fallback)
# ═══════════════════════════════════════════════════════════════
def _forecast_naive(serie: pd.DataFrame, horizonte: int) -> ForecastResult:
    """
    Seasonal Naive: el pronóstico de cada mes = el mismo mes del año anterior.
    Fórmula: ŷ_{T+h} = y_{T+h-s}  donde s=12 (mensual).

    Es el baseline estándar en competencias de pronóstico (M4, M5).
    Si un modelo no supera Seasonal Naive, no debe usarse.
    """
    y  = serie["y"].values.astype(float)
    ds = pd.to_datetime(serie["ds"].values)
    n  = len(y)
    s  = 12

    # Backtesting: últimos 6 meses
    n_test = min(6, n // 4)
    backtest_df, metricas = pd.DataFrame(), {}

    if n_test > 0 and n >= s + n_test:
        p_bt = np.array([y[n - n_test - s + i] for i in range(n_test)])
        p_bt = np.clip(p_bt, 0, None)
        backtest_df = pd.DataFrame({
            "ds": ds[-n_test:],
            "y_real": y[-n_test:],
            "y_pred": p_bt,
        })
        metricas = _metricas(y[-n_test:], p_bt)

    # Forecast
    fc_vals = []
    for h in range(1, horizonte + 1):
        idx = n - s + ((h - 1) % s)
        fc_vals.append(max(0.0, float(y[idx])) if idx >= 0 else 0.0)

    # Intervalo: ±1 std de la serie histórica proporcional a h
    sigma = float(np.std(y))
    z     = 1.645
    lower = [max(0.0, v - z * sigma * np.sqrt(i + 1)) for i, v in enumerate(fc_vals)]
    upper = [v + z * sigma * np.sqrt(i + 1)            for i, v in enumerate(fc_vals)]

    forecast_df = _build_hist_fc(serie, fc_vals, lower, upper)
    return ForecastResult(
        modelo="Naive Estacional (s=12)",
        historico=serie, forecast=forecast_df,
        metricas=metricas, backtest=backtest_df,
    )


# ═══════════════════════════════════════════════════════════════
# Punto de entrada principal
# ═══════════════════════════════════════════════════════════════
MODELOS_DISPONIBLES = {
    "sarima": "SARIMA (recomendado)",
    "auto":   "Automatico (mejor MAPE en backtesting)",
    "ets":    "Holt-Winters ETS",
    "xgb":    "XGBoost (ML)",
    "naive":  "Naive Estacional",
}

# Fecha de inicio del regimen actual de demanda (descendente y estabilizado)
# Solo SARIMA usa este recorte para capturar el comportamiento reciente
SARIMA_FECHA_INICIO = "2022-01-01"


def generar_forecast(
    df: pd.DataFrame,
    horizonte: int,
    col_periodo: str = "PERIODO",
    col_val: str = "PESO_TON",
    modelo: str = "auto",
) -> ForecastResult:
    """
    Genera un pronóstico de demanda mensual.

    Args:
        df:         DataFrame con col_periodo y col_val.
        horizonte:  Número de meses a pronosticar.
        modelo:     "auto" | "ets" | "sarima" | "xgb" | "naive"
                    En modo "auto" elige el de menor MAPE en backtesting.

    Returns:
        ForecastResult con modelo, forecast, métricas y backtest.
    """
    serie = _preparar_serie(df, col_periodo, col_val)

    if len(serie) < MIN_PERIODS_FORECAST:
        return ForecastResult(
            modelo="N/A", historico=serie,
            forecast=pd.DataFrame(), metricas={}, backtest=pd.DataFrame(),
            error_msg=(
                f"Serie insuficiente: {len(serie)} períodos. "
                f"Se requieren al menos {MIN_PERIODS_FORECAST}."
            ),
        )

    # Mapa de funciones
    _fns = {
        "ets":    _forecast_ets,
        "sarima": _forecast_sarima,
        "xgb":    _forecast_xgboost,
        "naive":  _forecast_naive,
    }

    if modelo in _fns:
        return _fns[modelo](serie, horizonte)

    # ── Modo AUTO: probar todos, elegir el de menor MAPE ─────
    candidatos_orden = ["ets", "xgb", "sarima", "naive"]
    resultados = []

    for key in candidatos_orden:
        r = _fns[key](serie, horizonte)
        if r.error_msg or r.forecast.empty:
            continue
        mape = r.metricas.get("MAPE (%)", float("inf"))
        if mape is None or np.isnan(mape):
            mape = float("inf")
        resultados.append((mape, key, r))

    if not resultados:
        # Último recurso: naive siempre funciona
        return _forecast_naive(serie, horizonte)

    resultados.sort(key=lambda x: x[0])
    mejor_mape, mejor_key, mejor_res = resultados[0]

    # Anotar en el nombre del modelo el proceso de selección
    ranking = " | ".join(
        f"{k}: {m:.0f}%" if m < float("inf") else f"{k}: N/A"
        for m, k, _ in resultados
    )
    mejor_res.modelo = f"{mejor_res.modelo}  ✓ [auto: {ranking}]"
    return mejor_res


# ─────────────────────────────────────────────
# Utilidades para la página
# ─────────────────────────────────────────────
def filtrar_por_dimension(
    df: pd.DataFrame,
    col_dim: str,
    valor: str,
    col_periodo: str = "PERIODO",
    col_val: str = "PESO_TON",
) -> pd.DataFrame:
    if df.empty or col_dim not in df.columns:
        return pd.DataFrame()
    filtrado = df[df[col_dim] == valor]
    if filtrado.empty:
        return pd.DataFrame()
    return (
        filtrado.groupby(col_periodo, as_index=False)[col_val]
        .sum().sort_values(col_periodo).reset_index(drop=True)
    )


def generar_forecast_multiple(
    df: pd.DataFrame,
    col_dim: str,
    horizonte: int,
    col_periodo: str = "PERIODO",
    col_val: str = "PESO_TON",
    top_n: int | None = None,
    modelo: str = "auto",
) -> dict:
    if df.empty or col_dim not in df.columns:
        return {}
    dims = df[col_dim].dropna().unique().tolist()
    if top_n:
        vol = df.groupby(col_dim)[col_val].sum().nlargest(top_n).index.tolist()
        dims = [d for d in dims if d in vol]
    resultados = {}
    for dim in dims:
        sub = (
            df[df[col_dim] == dim]
            .groupby(col_periodo, as_index=False)[col_val]
            .sum().sort_values(col_periodo).reset_index(drop=True)
        )
        if len(sub) >= MIN_PERIODS_FORECAST:
            resultados[dim] = generar_forecast(sub, horizonte, col_periodo, col_val, modelo)
    return resultados
