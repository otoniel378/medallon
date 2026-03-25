"""
04_Forecasting.py — Pronosticos de demanda mensual.
Modelos: Holt-Winters ETS | SARIMA | XGBoost | Naive Estacional | Auto
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from config import APP_NAME, COLORS, COLOR_SEQUENCE, FORECAST_HORIZON_DEFAULT, FORECAST_HORIZON_MAX

from data.loaders import (
    load_gold_demanda_mensual_total,
    load_gold_demanda_mensual,
    load_serie_mensual_proceso,
)
from analytics.forecasting import (
    generar_forecast,
    filtrar_por_dimension,
    MODELOS_DISPONIBLES,
)
from components.kpi_cards import seccion_titulo
from components.filters import sidebar_header
from components.tables import tabla_ejecutiva, tabla_metricas

st.set_page_config(
    page_title=f"Forecasting — {APP_NAME}",
    page_icon="🔮",
    layout="wide",
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
sidebar_header("Parametros", "🔮")

horizonte = st.sidebar.slider(
    "Horizonte (meses)", min_value=1,
    max_value=FORECAST_HORIZON_MAX, value=FORECAST_HORIZON_DEFAULT,
    key="fc_horizonte",
)

modelo_key = st.sidebar.selectbox(
    "Modelo de pronostico",
    options=list(MODELOS_DISPONIBLES.keys()),
    format_func=lambda k: MODELOS_DISPONIBLES[k],
    index=0,   # sarima es el primero y el default
    key="fc_modelo",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style='font-size:0.78rem;color:{COLORS["text_light"]};'>
    <b>Guia de modelos</b><br><br>
    ETS — Holt-Winters. Ideal para demanda estable con estacionalidad anual.<br><br>
    SARIMA — Clasico estadistico. Bueno cuando hay tendencia clara.<br><br>
    XGBoost — Machine Learning con rezagos. Captura patrones no lineales.<br><br>
    Naive — Baseline: igual al mismo mes del anio pasado. Simple y robusto.<br><br>
    Auto — Prueba los 4 y elige el de menor MAPE en backtesting.
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────
with st.spinner("Cargando datos..."):
    df_total   = load_gold_demanda_mensual_total()
    df_mensual = load_gold_demanda_mensual()
    df_proceso = load_serie_mensual_proceso()

# ─────────────────────────────────────────────
# Encabezado
# ─────────────────────────────────────────────
st.markdown(
    f"""
    <h2 style='color:{COLORS["primary"]};margin-bottom:0;'>Forecasting de Demanda</h2>
    <p style='color:{COLORS["text_light"]};'>
        Modelo: <b>{MODELOS_DISPONIBLES[modelo_key]}</b> &nbsp;&middot;&nbsp; Horizonte: <b>{horizonte} meses</b>
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()


# ─────────────────────────────────────────────
# Helpers de visualizacion
# ─────────────────────────────────────────────
def _colores_modelo(nombre: str) -> str:
    if "ETS" in nombre or "Holt" in nombre:
        return COLORS["success"]
    if "XGB" in nombre:
        return COLOR_SEQUENCE[1]
    if "SARIMA" in nombre or "ARIMA" in nombre:
        return COLORS["secondary"]
    return COLORS["neutral"]


def _grafico_forecast(resultado, titulo="") -> go.Figure:
    fc_df, hist_df = resultado.forecast, resultado.historico
    fig = go.Figure()
    if fc_df.empty or hist_df.empty:
        return fig

    fc_fut = fc_df[fc_df["ds"] > hist_df["ds"].max()].copy()
    color_fc = _colores_modelo(resultado.modelo)

    if not fc_fut.empty and fc_fut["yhat_upper"].notna().any():
        band = fc_fut.dropna(subset=["yhat_upper", "yhat_lower"])
        if not band.empty:
            fig.add_trace(go.Scatter(
                x=pd.concat([band["ds"], band["ds"].iloc[::-1]]),
                y=pd.concat([band["yhat_upper"], band["yhat_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(74,123,167,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Banda 90%", hoverinfo="skip",
            ))

    fig.add_trace(go.Scatter(
        x=hist_df["ds"], y=hist_df["y"],
        name="Historico",
        line=dict(color=COLORS["primary"], width=2.5),
        hovertemplate="%{x|%b %Y}<br>Real: %{y:,.1f} ton<extra></extra>",
    ))

    if not fc_fut.empty:
        fig.add_trace(go.Scatter(
            x=fc_fut["ds"], y=fc_fut["yhat"],
            name=f"Pronostico ({horizonte}m)",
            line=dict(color=color_fc, width=2.5, dash="dot"),
            mode="lines+markers", marker=dict(size=7),
            hovertemplate="%{x|%b %Y}<br>Pronostico: %{y:,.1f} ton<extra></extra>",
        ))

    corte = str(hist_df["ds"].max())
    fig.add_shape(type="line", x0=corte, x1=corte, y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color=COLORS["neutral"], width=1.5, dash="dot"))
    fig.add_annotation(x=corte, y=1, xref="x", yref="paper",
                       text="Hoy", showarrow=False, yanchor="bottom",
                       font=dict(size=11, color=COLORS["neutral"]))
    fig.update_layout(
        paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"]),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(showgrid=False), yaxis=dict(title="Toneladas", gridcolor="#E5E7EB"),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        title=dict(text=titulo, font=dict(size=14, color=COLORS["primary"]), x=0),
        height=390,
    )
    return fig


def _tabla_futuro(resultado) -> pd.DataFrame:
    fc, hist = resultado.forecast, resultado.historico
    if fc.empty or hist.empty:
        return pd.DataFrame()
    fut = fc[fc["ds"] > hist["ds"].max()].copy()
    fut = fut.rename(columns={
        "ds": "PERIODO", "yhat": "FORECAST_TON",
        "yhat_lower": "LOWER_90", "yhat_upper": "UPPER_90",
    })
    fut["FORECAST_TON"] = fut["FORECAST_TON"].round(1)
    if "LOWER_90" in fut.columns:
        fut["LOWER_90"] = fut["LOWER_90"].round(1)
    if "UPPER_90" in fut.columns:
        fut["UPPER_90"] = fut["UPPER_90"].round(1)
    return fut.reset_index(drop=True)


def _render_resultado(res, key_prefix: str):
    """Renderiza modelo, metricas, grafico, tabla y backtesting."""
    if res.error_msg:
        st.error(f"Error: {res.error_msg}")
        return

    color_mod = _colores_modelo(res.modelo)

    col_m, col_mt = st.columns([1, 3])
    with col_m:
        st.markdown(
            f"""<div style='background:{COLORS["surface"]};border:1px solid #E5E7EB;
            border-left:5px solid {color_mod};border-radius:8px;padding:14px;'>
            <div style='color:{COLORS["text_light"]};font-size:0.75rem;font-weight:600;'>MODELO USADO</div>
            <div style='color:{color_mod};font-size:1.05rem;font-weight:700;line-height:1.4;'>{res.modelo}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_mt:
        mape = res.metricas.get("MAPE (%)", float("nan")) if res.metricas else float("nan")
        mape_txt = f"{mape:.1f}%" if not np.isnan(mape) else "N/A"
        if not np.isnan(mape):
            if mape > 50:
                nivel = "alto"
                color_mape = COLORS.get("danger", "#C0392B")
            elif mape > 30:
                nivel = "moderado"
                color_mape = COLORS.get("warning", "#D4A017")
            else:
                nivel = "aceptable"
                color_mape = COLORS.get("success", "#2D8A5C")
            st.markdown(
                f"<div style='background:{color_mape}22;border:1px solid {color_mape}55;"
                f"border-radius:6px;padding:8px 14px;color:{color_mape};font-weight:600;'>"
                f"MAPE {nivel}: {mape_txt}</div>",
                unsafe_allow_html=True,
            )
        if res.metricas:
            tabla_metricas(res.metricas, titulo="Metricas de backtesting")

    fig = _grafico_forecast(res, titulo=f"Historico + Pronostico {horizonte} meses")
    st.plotly_chart(fig, use_container_width=True)

    df_fut = _tabla_futuro(res)
    if not df_fut.empty:
        seccion_titulo(f"Valores pronosticados - proximos {horizonte} meses")
        tabla_ejecutiva(
            df_fut,
            col_formatos={"FORECAST_TON": "{:,.1f}", "LOWER_90": "{:,.1f}", "UPPER_90": "{:,.1f}"},
            key=f"{key_prefix}_tabla",
            height=270,
        )

    if not res.backtest.empty:
        st.divider()
        seccion_titulo("Backtesting", "Comparacion real vs predicho")
        bt = res.backtest.copy()
        bt["ERROR_ABS"] = (bt["y_real"] - bt["y_pred"]).abs().round(1)
        bt["ERROR_PCT"] = (bt["ERROR_ABS"] / bt["y_real"].replace(0, np.nan) * 100).round(1)
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Bar(x=bt["ds"], y=bt["y_real"], name="Real",
                                 marker_color=COLORS["primary"], opacity=0.8))
        fig_bt.add_trace(go.Scatter(x=bt["ds"], y=bt["y_pred"], name="Predicho",
                                     mode="lines+markers",
                                     line=dict(color=color_mod, width=2.5, dash="dot"),
                                     marker=dict(size=9)))
        fig_bt.update_layout(
            paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["background"],
            font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"]),
            margin=dict(l=40, r=20, t=30, b=40), height=280,
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#E5E7EB", title="Toneladas"),
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
            barmode="overlay",
        )
        st.plotly_chart(fig_bt, use_container_width=True)


# ─────────────────────────────────────────────
# Cache de resultados en session_state
# ─────────────────────────────────────────────
def _cache_key(prefix: str, modelo: str, horizonte: int, dim: str = "") -> str:
    return f"fc_{prefix}_{modelo}_{horizonte}_{dim}"


def _get_or_compute(cache_key: str, fn):
    """Retorna resultado cacheado o ejecuta fn() y lo guarda."""
    if cache_key not in st.session_state:
        with st.spinner("Calculando pronostico..."):
            st.session_state[cache_key] = fn()
    return st.session_state[cache_key]


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Demanda Total",
    "Comparar Modelos",
    "Por Proceso",
    "Por Familia",
])

# ══════════════════════════════════════════════
# TAB 1 — Demanda Total
# ══════════════════════════════════════════════
with tab1:
    seccion_titulo("Demanda Total", f"{horizonte} meses proyectados")

    if df_total.empty:
        st.warning("Sin datos de demanda total.")
    else:
        ck = _cache_key("total", modelo_key, horizonte)
        res_total = _get_or_compute(ck, lambda: generar_forecast(df_total, horizonte, modelo=modelo_key))
        _render_resultado(res_total, key_prefix="total")

# ══════════════════════════════════════════════
# TAB 2 — Comparar los 4 modelos lado a lado
# ══════════════════════════════════════════════
with tab2:
    seccion_titulo("Comparacion de Modelos", "Ejecuta los 4 modelos y compara MAPE, MAE y RMSE")

    if df_total.empty:
        st.warning("Sin datos.")
    else:
        if st.button("Ejecutar comparacion de los 4 modelos", key="btn_comparar"):
            modelos_eval = ["ets", "sarima", "xgb", "naive"]
            resultados_comp = {}
            prog = st.progress(0)
            for i, mk in enumerate(modelos_eval):
                ck_c = _cache_key("comp", mk, horizonte)
                if ck_c not in st.session_state:
                    st.session_state[ck_c] = generar_forecast(df_total, horizonte, modelo=mk)
                resultados_comp[mk] = st.session_state[ck_c]
                prog.progress((i + 1) / len(modelos_eval))
            prog.empty()

            rows = []
            for mk, r in resultados_comp.items():
                if r.error_msg:
                    rows.append({"Modelo": r.modelo, "MAE": "—", "MAPE": "—", "RMSE": "—"})
                else:
                    m = r.metricas
                    mape_v = m.get("MAPE (%)", float("nan"))
                    rows.append({
                        "Modelo": MODELOS_DISPONIBLES[mk],
                        "MAE":    m.get("MAE", "—"),
                        "MAPE":   f"{mape_v:.1f}%" if not np.isnan(mape_v) else "—",
                        "RMSE":   m.get("RMSE", "—"),
                    })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            fig_comp = go.Figure()
            colores_comp = {
                "ets":    COLORS["success"],
                "sarima": COLORS["secondary"],
                "xgb":    COLOR_SEQUENCE[1],
                "naive":  COLORS["neutral"],
            }
            hist = resultados_comp["ets"].historico
            if not hist.empty:
                fig_comp.add_trace(go.Scatter(
                    x=hist["ds"], y=hist["y"], name="Historico",
                    line=dict(color=COLORS["primary"], width=2.5),
                    hovertemplate="%{x|%b %Y}: %{y:,.1f} ton<extra></extra>",
                ))
            for mk, r in resultados_comp.items():
                if r.error_msg or r.forecast.empty:
                    continue
                fc_fut = r.forecast[r.forecast["ds"] > hist["ds"].max()]
                if fc_fut.empty:
                    continue
                fig_comp.add_trace(go.Scatter(
                    x=fc_fut["ds"], y=fc_fut["yhat"],
                    name=MODELOS_DISPONIBLES[mk],
                    line=dict(color=colores_comp[mk], width=2, dash="dot"),
                    mode="lines+markers", marker=dict(size=6),
                    hovertemplate=f"<b>{MODELOS_DISPONIBLES[mk]}</b><br>%{{x|%b %Y}}: %{{y:,.1f}} ton<extra></extra>",
                ))
            fig_comp.update_layout(
                paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["background"],
                font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"]),
                margin=dict(l=40, r=20, t=50, b=40),
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#E5E7EB", title="Toneladas"),
                legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
                title=dict(text="Comparacion de pronosticos — todos los modelos",
                           font=dict(size=14, color=COLORS["primary"]), x=0),
                height=420,
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Haz clic en el boton para comparar los 4 modelos en la demanda total.")

# ══════════════════════════════════════════════
# TAB 3 — Por Proceso
# ══════════════════════════════════════════════
with tab3:
    seccion_titulo("Pronostico por Proceso", f"Modelo: {MODELOS_DISPONIBLES[modelo_key]}")

    if df_proceso.empty:
        st.warning("Sin datos de serie por proceso.")
    else:
        procesos = sorted(df_proceso["PROCESO"].dropna().unique().tolist())
        proc_sel = st.selectbox("Proceso:", options=procesos, key="fc_proc")

        if proc_sel:
            df_p = filtrar_por_dimension(df_proceso, "PROCESO", proc_sel)
            n_p  = len(df_p)
            st.caption(f"Serie disponible: **{n_p} meses**")

            if n_p < 12:
                st.warning(f"Solo {n_p} meses disponibles — se requieren minimo 12.")
            else:
                ck_p = _cache_key("proc", modelo_key, horizonte, proc_sel)
                res_p = _get_or_compute(ck_p, lambda: generar_forecast(df_p, horizonte, modelo=modelo_key))
                _render_resultado(res_p, key_prefix="proc")

        st.divider()
        with st.expander("Resumen comparativo de todos los procesos"):
            if st.button("Calcular pronostico para todos los procesos", key="btn_all_proc"):
                res_procs = {}
                prog2 = st.progress(0)
                for i, proc in enumerate(procesos):
                    df_pi = filtrar_por_dimension(df_proceso, "PROCESO", proc)
                    if len(df_pi) >= 12:
                        ck_pi = _cache_key("proc", modelo_key, horizonte, proc)
                        if ck_pi not in st.session_state:
                            st.session_state[ck_pi] = generar_forecast(df_pi, horizonte, modelo=modelo_key)
                        r = st.session_state[ck_pi]
                        if not r.error_msg:
                            fut = _tabla_futuro(r)
                            if not fut.empty:
                                res_procs[proc] = {
                                    "forecast_total": fut["FORECAST_TON"].sum(),
                                    "mape": r.metricas.get("MAPE (%)", None),
                                }
                    prog2.progress((i + 1) / len(procesos))
                prog2.empty()

                if res_procs:
                    df_rp = pd.DataFrame([
                        {"PROCESO": k, "FORECAST_TON": v["forecast_total"], "MAPE": v["mape"]}
                        for k, v in res_procs.items()
                    ]).sort_values("FORECAST_TON", ascending=True)

                    fig_rp = go.Figure(go.Bar(
                        x=df_rp["FORECAST_TON"], y=df_rp["PROCESO"],
                        orientation="h",
                        marker=dict(color=df_rp["FORECAST_TON"],
                                    colorscale=[[0, COLORS["accent"]], [1, COLORS["primary"]]]),
                        text=df_rp["FORECAST_TON"].round(0), textposition="outside",
                        hovertemplate="<b>%{y}</b><br>Forecast: %{x:,.1f} ton<extra></extra>",
                    ))
                    fig_rp.update_layout(
                        paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["background"],
                        font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"]),
                        margin=dict(l=10, r=80, t=50, b=40),
                        xaxis=dict(title=f"Forecast acumulado {horizonte}m (ton)", gridcolor="#E5E7EB"),
                        showlegend=False,
                        height=max(300, len(df_rp) * 38 + 80),
                        title=dict(text=f"Demanda esperada por proceso — proximos {horizonte} meses",
                                   font=dict(size=13, color=COLORS["primary"]), x=0),
                    )
                    st.plotly_chart(fig_rp, use_container_width=True)
                    tabla_ejecutiva(
                        df_rp.sort_values("FORECAST_TON", ascending=False),
                        col_formatos={"FORECAST_TON": "{:,.1f}", "MAPE": "{:.1f}%"},
                        key="resumen_procs",
                        height=260,
                    )

# ══════════════════════════════════════════════
# TAB 4 — Por Familia (Producto)
# ══════════════════════════════════════════════
with tab4:
    seccion_titulo("Pronostico por Familia de Producto", f"Modelo: {MODELOS_DISPONIBLES[modelo_key]}")

    excluir_fc = {"OTROS", "OTHER", "N/D", "SIN CLASIFICAR", "S/C"}

    if df_mensual.empty or "PRODUCTO_LIMPIO" not in df_mensual.columns:
        st.warning("Sin datos granulares por producto.")
    else:
        prods = sorted([
            p for p in df_mensual["PRODUCTO_LIMPIO"].dropna().unique()
            if str(p).upper() not in excluir_fc
        ])
        prod_sel = st.selectbox("Producto:", options=prods, key="fc_prod")

        if prod_sel:
            df_pr = filtrar_por_dimension(df_mensual, "PRODUCTO_LIMPIO", prod_sel)
            n_pr  = len(df_pr)
            st.caption(f"Serie disponible: **{n_pr} meses**")

            if n_pr < 12:
                st.warning(f"Solo {n_pr} meses — se requieren minimo 12.")
            else:
                ck_pr = _cache_key("prod", modelo_key, horizonte, prod_sel)
                res_pr = _get_or_compute(ck_pr, lambda: generar_forecast(df_pr, horizonte, modelo=modelo_key))
                _render_resultado(res_pr, key_prefix="prod")

        st.divider()
        with st.expander("Resumen comparativo de todas las familias"):
            if st.button("Calcular pronostico para todas las familias", key="btn_all_prod"):
                res_fams = {}
                prog3 = st.progress(0)
                for i, prod in enumerate(prods):
                    df_pri = filtrar_por_dimension(df_mensual, "PRODUCTO_LIMPIO", prod)
                    if len(df_pri) >= 12:
                        ck_pri = _cache_key("prod", modelo_key, horizonte, prod)
                        if ck_pri not in st.session_state:
                            st.session_state[ck_pri] = generar_forecast(df_pri, horizonte, modelo=modelo_key)
                        r = st.session_state[ck_pri]
                        if not r.error_msg:
                            fut = _tabla_futuro(r)
                            if not fut.empty:
                                res_fams[prod] = {
                                    "forecast_total": fut["FORECAST_TON"].sum(),
                                    "mape": r.metricas.get("MAPE (%)", None),
                                }
                    prog3.progress((i + 1) / len(prods))
                prog3.empty()

                if res_fams:
                    df_rf = pd.DataFrame([
                        {"FAMILIA": k, "FORECAST_TON": v["forecast_total"], "MAPE": v["mape"]}
                        for k, v in res_fams.items()
                    ]).sort_values("FORECAST_TON", ascending=True)

                    fig_rf = px.bar(
                        df_rf, x="FORECAST_TON", y="FAMILIA", orientation="h",
                        color="FORECAST_TON",
                        color_continuous_scale=[[0, COLORS["accent"]], [1, COLORS["primary"]]],
                        text=df_rf["FORECAST_TON"].round(0),
                        labels={"FORECAST_TON": f"Forecast {horizonte}m (ton)", "FAMILIA": ""},
                        title=f"Demanda esperada por familia — proximos {horizonte} meses",
                    )
                    fig_rf.update_traces(textposition="outside")
                    fig_rf.update_layout(
                        paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["background"],
                        font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"]),
                        margin=dict(l=10, r=80, t=50, b=40),
                        xaxis=dict(gridcolor="#E5E7EB"),
                        coloraxis_showscale=False,
                        height=max(300, len(df_rf) * 38 + 80),
                        title=dict(font=dict(size=13, color=COLORS["primary"]), x=0),
                    )
                    st.plotly_chart(fig_rf, use_container_width=True)
                    tabla_ejecutiva(
                        df_rf.sort_values("FORECAST_TON", ascending=False),
                        col_formatos={"FORECAST_TON": "{:,.1f}", "MAPE": "{:.1f}%"},
                        key="resumen_fams",
                        height=260,
                    )
