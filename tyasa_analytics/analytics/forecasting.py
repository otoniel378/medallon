"""
forecasting.py — Pronósticos de demanda mensual.
Soporta Prophet (preferido) y ARIMA como fallback.
Incluye backtesting y métricas de error.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from config import MIN_PERIODS_FORECAST


@dataclass
class ForecastResult:
    """Contenedor de resultados del forecasting."""
    modelo: str
    historico: pd.DataFrame       # ds, y
    forecast: pd.DataFrame        # ds, yhat, yhat_lower, yhat_upper
    metricas: dict                # MAE, MAPE, RMSE
    backtest: pd.DataFrame        # ds, y_real, y_pred
    error_msg: str | None = None


def _metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula MAE, MAPE y RMSE."""
    y_real = np.array(y_real, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_real - y_pred)))
    rmse = float(np.sqrt(np.mean((y_real - y_pred) ** 2)))

    mask = y_real != 0
    mape = float(np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100) if mask.sum() > 0 else float("nan")

    return {
        "MAE": round(mae, 2),
        "MAPE (%)": round(mape, 1),
        "RMSE": round(rmse, 2),
    }


def _preparar_serie(df: pd.DataFrame, col_periodo: str = "PERIODO", col_val: str = "PESO_TON") -> pd.DataFrame:
    """Convierte DataFrame a formato ds/y para Prophet."""
    serie = df[[col_periodo, col_val]].copy()
    serie.columns = ["ds", "y"]
    serie["ds"] = pd.to_datetime(serie["ds"])
    serie["y"] = pd.to_numeric(serie["y"], errors="coerce").fillna(0)
    serie = serie.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    return serie


def _forecast_prophet(serie: pd.DataFrame, horizonte: int) -> ForecastResult:
    """Genera forecast con Prophet."""
    from prophet import Prophet
    import logging
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    # Backtesting: últimos 3 meses como test
    n_test = min(3, len(serie) // 4)
    train = serie.iloc[:-n_test] if n_test > 0 else serie
    test = serie.iloc[-n_test:] if n_test > 0 else pd.DataFrame()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.9,
    )
    model.fit(train)

    # Backtest predictions
    backtest_df = pd.DataFrame()
    metricas = {}
    if not test.empty:
        future_bt = model.make_future_dataframe(periods=n_test, freq="MS")
        fc_bt = model.predict(future_bt)
        preds_bt = fc_bt.tail(n_test)[["ds", "yhat"]].copy()
        preds_bt = preds_bt.merge(test[["ds", "y"]], on="ds", how="inner")
        backtest_df = preds_bt.rename(columns={"y": "y_real", "yhat": "y_pred"})
        if not backtest_df.empty:
            metricas = _metricas(backtest_df["y_real"].values, backtest_df["y_pred"].values)

    # Forecast completo
    model_full = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.9,
    )
    model_full.fit(serie)
    future = model_full.make_future_dataframe(periods=horizonte, freq="MS")
    fc = model_full.predict(future)

    forecast_df = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_df["yhat"] = forecast_df["yhat"].clip(lower=0)
    forecast_df["yhat_lower"] = forecast_df["yhat_lower"].clip(lower=0)

    return ForecastResult(
        modelo="Prophet",
        historico=serie,
        forecast=forecast_df,
        metricas=metricas,
        backtest=backtest_df,
    )


def _forecast_arima(serie: pd.DataFrame, horizonte: int) -> ForecastResult:
    """Genera forecast con SARIMA/ARIMA (statsmodels)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import warnings

    y = serie["y"].values
    n = len(y)

    # Elegir orden estacional según cantidad de datos
    if n >= 36:
        seasonal_order = (1, 1, 0, 12)
        modelo_nombre = "SARIMA(1,1,1)(1,1,0,12)"
    elif n >= 24:
        seasonal_order = (1, 0, 0, 12)
        modelo_nombre = "SARIMA(1,1,1)(1,0,0,12)"
    else:
        seasonal_order = (0, 0, 0, 0)
        modelo_nombre = "ARIMA(1,1,1)"

    # Backtesting: últimos 3 meses
    n_test = min(3, n // 4)
    y_train = y[:-n_test] if n_test > 0 else y
    y_test = y[-n_test:] if n_test > 0 else np.array([])

    backtest_df = pd.DataFrame()
    metricas = {}

    def _fit(y_arr, s_order):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m = SARIMAX(
                    y_arr, order=(1, 1, 1), seasonal_order=s_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False, maxiter=200)
                return m
            except Exception:
                from statsmodels.tsa.arima.model import ARIMA as _ARIMA
                return _ARIMA(y_arr, order=(1, 1, 1)).fit()

    try:
        if n_test > 0:
            model_bt = _fit(y_train, seasonal_order)
            preds_bt = model_bt.forecast(steps=n_test)
            bt_dates = serie["ds"].iloc[-n_test:].values
            backtest_df = pd.DataFrame({
                "ds": bt_dates,
                "y_real": y_test,
                "y_pred": np.clip(preds_bt, 0, None),
            })
            metricas = _metricas(y_test, preds_bt)

        # Forecast completo
        model_full = _fit(y, seasonal_order)
        fc_obj = model_full.get_forecast(steps=horizonte)

        # summary_frame() siempre devuelve DataFrame (evita problemas con ndarray)
        try:
            fc_frame = fc_obj.summary_frame(alpha=0.10)
            fc_mean = np.clip(fc_frame["mean"].values, 0, None)
            lower = np.clip(fc_frame["mean_ci_lower"].values, 0, None)
            upper = fc_frame["mean_ci_upper"].values
        except Exception:
            # Fallback manual si summary_frame falla
            raw_mean = np.asarray(fc_obj.predicted_mean).ravel()
            fc_mean = np.clip(raw_mean, 0, None)
            lower = fc_mean * 0.85
            upper = fc_mean * 1.15

        last_date = serie["ds"].max()
        future_dates = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1),
            periods=horizonte,
            freq="MS",
        )

        # Histórico con banda nula + forecast con banda
        hist_part = serie[["ds", "y"]].rename(columns={"y": "yhat"})
        hist_part["yhat_lower"] = np.nan
        hist_part["yhat_upper"] = np.nan

        fc_df = pd.DataFrame({
            "ds": future_dates,
            "yhat": fc_mean,
            "yhat_lower": lower,
            "yhat_upper": upper,
        })

        forecast_df = pd.concat([hist_part, fc_df], ignore_index=True)

        return ForecastResult(
            modelo=modelo_nombre,
            historico=serie,
            forecast=forecast_df,
            metricas=metricas,
            backtest=backtest_df,
        )

    except Exception as e:
        return ForecastResult(
            modelo="ARIMA",
            historico=serie,
            forecast=pd.DataFrame(),
            metricas={},
            backtest=pd.DataFrame(),
            error_msg=str(e),
        )


def generar_forecast(
    df: pd.DataFrame,
    horizonte: int,
    col_periodo: str = "PERIODO",
    col_val: str = "PESO_TON",
) -> ForecastResult:
    """
    Punto de entrada principal para generar un forecast.
    Intenta Prophet primero; usa ARIMA si Prophet no está disponible.

    Args:
        df: DataFrame con col_periodo y col_val.
        horizonte: número de meses a proyectar.
        col_periodo: nombre de la columna de periodo.
        col_val: nombre de la columna de valor.

    Returns:
        ForecastResult con modelo, forecast, métricas y backtest.
    """
    serie = _preparar_serie(df, col_periodo, col_val)

    if len(serie) < MIN_PERIODS_FORECAST:
        return ForecastResult(
            modelo="N/A",
            historico=serie,
            forecast=pd.DataFrame(),
            metricas={},
            backtest=pd.DataFrame(),
            error_msg=(
                f"Serie insuficiente: {len(serie)} periodos. "
                f"Se requieren al menos {MIN_PERIODS_FORECAST} para hacer forecast."
            ),
        )

    # Intentar Prophet
    try:
        import prophet  # noqa
        return _forecast_prophet(serie, horizonte)
    except ImportError:
        pass

    # Fallback: ARIMA
    return _forecast_arima(serie, horizonte)


def generar_forecast_multiple(
    df: pd.DataFrame,
    col_dim: str,
    horizonte: int,
    col_periodo: str = "PERIODO",
    col_val: str = "PESO_TON",
    top_n: int | None = None,
) -> dict:
    """
    Genera forecast para múltiples dimensiones (procesos o productos).

    Args:
        df: DataFrame con col_dim, col_periodo, col_val.
        col_dim: columna de agrupación.
        horizonte: meses a pronosticar.
        top_n: si se especifica, toma las top_n dimensiones por volumen.

    Returns:
        dict {dimension: ForecastResult} para dimensiones con datos suficientes.
    """
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
            .sum()
            .sort_values(col_periodo)
            .reset_index(drop=True)
        )
        if len(sub) >= MIN_PERIODS_FORECAST:
            resultados[dim] = generar_forecast(sub, horizonte, col_periodo, col_val)
    return resultados


def filtrar_por_dimension(
    df: pd.DataFrame,
    col_dim: str,
    valor: str,
    col_periodo: str = "PERIODO",
    col_val: str = "PESO_TON",
) -> pd.DataFrame:
    """
    Filtra y agrega el DataFrame por una dimensión específica (cliente o producto).

    Args:
        df: DataFrame con col_dim, col_periodo, col_val.
        col_dim: columna de dimensión.
        valor: valor a filtrar.
        col_periodo: columna de periodo.
        col_val: columna de valor.

    Returns:
        Serie mensual agregada para el valor seleccionado.
    """
    if df.empty or col_dim not in df.columns:
        return pd.DataFrame()

    filtrado = df[df[col_dim] == valor].copy()
    if filtrado.empty:
        return pd.DataFrame()

    agrupado = (
        filtrado.groupby(col_periodo, as_index=False)[col_val]
        .sum()
        .sort_values(col_periodo)
        .reset_index(drop=True)
    )
    return agrupado
