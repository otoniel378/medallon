"""
04_Forecasting.py — Pronosticos de demanda mensual con SARIMA/ARIMA.
Demanda total, por proceso y por familia de producto.
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
from analytics.forecasting import generar_forecast, filtrar_por_dimension
from components.kpi_cards import seccion_titulo
from components.filters import sidebar_header
from components.tables import tabla_ejecutiva, tabla_metricas

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"Forecasting — {APP_NAME}",
    page_icon="🔮",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
sidebar_header("Parametros", "🔮")

horizonte = st.sidebar.slider(
    "Horizonte (meses)",
    min_value=1,
    max_value=FORECAST_HORIZON_MAX,
    value=FORECAST_HORIZON_DEFAULT,
    key="fc_horizonte",
)

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
with st.spinner("Cargando datos…"):
    df_total = load_gold_demanda_mensual_total()
    df_mensual = load_gold_demanda_mensual()
    df_proceso = load_serie_mensual_proceso()

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <h2 style='color:{COLORS["primary"]};margin-bottom:0;'>🔮 Forecasting de Demanda</h2>
    <p style='color:{COLORS["text_light"]};'>Pronosticos mensuales — Horizonte: {horizonte} meses</p>
    """,
    unsafe_allow_html=True,
)
st.divider()


# ---------------------------------------------------------------------------
# Helper: grafico historico + forecast
# ---------------------------------------------------------------------------
def _grafico_forecast(resultado, titulo: str = "") -> go.Figure:
    """Construye figura de linea historico + forecast + banda."""
    fc_df = resultado.forecast
    hist_df = resultado.historico

    fig = go.Figure()

    if fc_df.empty or hist_df.empty:
        return fig

    # Banda de confianza (solo puntos futuros con banda valida)
    fc_fut = fc_df[fc_df["ds"] > hist_df["ds"].max()].copy()
    if not fc_fut.empty and "yhat_upper" in fc_fut.columns:
        mask_band = fc_fut["yhat_upper"].notna() & fc_fut["yhat_lower"].notna()
        fc_band = fc_fut[mask_band]
        if not fc_band.empty:
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_band["ds"], fc_band["ds"].iloc[::-1]]),
                y=pd.concat([fc_band["yhat_upper"], fc_band["yhat_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(74,123,167,0.18)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Banda 90%",
                hoverinfo="skip",
            ))

    # Historico real
    fig.add_trace(go.Scatter(
        x=hist_df["ds"], y=hist_df["y"],
        name="Historico",
        line=dict(color=COLORS["primary"], width=2.5),
        mode="lines",
        hovertemplate="%{x|%b %Y}<br>Real: %{y:,.1f} ton<extra></extra>",
    ))

    # Forecast futuro
    if not fc_fut.empty:
        fig.add_trace(go.Scatter(
            x=fc_fut["ds"], y=fc_fut["yhat"],
            name=f"Pronostico ({horizonte} meses)",
            line=dict(color=COLORS["warning"], width=2.5, dash="dot"),
            mode="lines+markers",
            marker=dict(size=7),
            hovertemplate="%{x|%b %Y}<br>Pronostico: %{y:,.1f} ton<extra></extra>",
        ))

    # Linea divisoria
    corte = str(hist_df["ds"].max())
    fig.add_shape(
        type="line", x0=corte, x1=corte, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=COLORS["neutral"], width=1.5, dash="dot"),
    )
    fig.add_annotation(
        x=corte, y=1, xref="x", yref="paper",
        text="Hoy", showarrow=False, yanchor="bottom",
        font=dict(size=11, color=COLORS["neutral"]),
    )

    fig.update_layout(
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"]),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(title="", showgrid=False),
        yaxis=dict(title="Toneladas", gridcolor="#E5E7EB"),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        title=dict(text=titulo, font=dict(size=14, color=COLORS["primary"]), x=0),
        height=380,
    )
    return fig


def _tabla_forecast_futuro(resultado) -> pd.DataFrame:
    """Extrae solo los periodos futuros del forecast como tabla."""
    fc = resultado.forecast
    hist = resultado.historico
    if fc.empty or hist.empty:
        return pd.DataFrame()
    fut = fc[fc["ds"] > hist["ds"].max()].copy()
    fut = fut.rename(columns={
        "ds": "PERIODO", "yhat": "FORECAST_TON",
        "yhat_lower": "LOWER_90", "yhat_upper": "UPPER_90",
    })
    fut["FORECAST_TON"] = fut["FORECAST_TON"].clip(lower=0).round(1)
    if "LOWER_90" in fut.columns:
        fut["LOWER_90"] = fut["LOWER_90"].clip(lower=0).round(1)
    if "UPPER_90" in fut.columns:
        fut["UPPER_90"] = fut["UPPER_90"].round(1)
    return fut.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tabs principales
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Demanda Total", "⚙️ Por Proceso", "📦 Por Familia"])

# ============================================================
# TAB 1 — Demanda Total
# ============================================================
with tab1:
    seccion_titulo(
        "Demanda Total — Pronostico",
        f"Historico completo + {horizonte} meses proyectados",
    )

    if df_total.empty:
        st.warning("Sin datos de demanda total disponibles.")
    else:
        with st.spinner("Calculando pronostico total…"):
            res_total = generar_forecast(df_total, horizonte=horizonte)

        if res_total.error_msg:
            st.error(f"No fue posible generar el pronostico: {res_total.error_msg}")
        else:
            # Info modelo + metricas
            col_mod, col_met = st.columns([1, 3])
            with col_mod:
                st.markdown(
                    f"""
                    <div style='background:{COLORS["surface"]};border:1px solid #E5E7EB;
                    border-left:4px solid {COLORS["primary"]};border-radius:8px;padding:16px;'>
                        <div style='color:{COLORS["text_light"]};font-size:0.78rem;font-weight:600;'>MODELO</div>
                        <div style='color:{COLORS["primary"]};font-size:1.3rem;font-weight:700;'>{res_total.modelo}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_met:
                if res_total.metricas:
                    tabla_metricas(res_total.metricas, titulo="Metricas de backtesting")
                else:
                    st.info("Sin metricas de backtesting (datos insuficientes).")

            st.divider()

            # Grafico principal
            fig_t = _grafico_forecast(res_total, titulo="Demanda mensual total + pronostico")
            st.plotly_chart(fig_t, width="stretch")

            # Tabla de valores futuros
            df_fut_t = _tabla_forecast_futuro(res_total)
            if not df_fut_t.empty:
                seccion_titulo(f"Tabla — Pronostico {horizonte} meses")
                cols_fmt = {"FORECAST_TON": "{:,.1f}"}
                if "LOWER_90" in df_fut_t.columns:
                    cols_fmt["LOWER_90"] = "{:,.1f}"
                if "UPPER_90" in df_fut_t.columns:
                    cols_fmt["UPPER_90"] = "{:,.1f}"
                tabla_ejecutiva(df_fut_t, col_formatos=cols_fmt, key="fc_total_tabla", height=280)

            # Backtesting
            if not res_total.backtest.empty:
                st.divider()
                seccion_titulo("Backtesting", "Comparacion real vs predicho en ultimos meses")
                bt = res_total.backtest.copy()
                bt["ERROR_ABS"] = (bt["y_real"] - bt["y_pred"]).abs().round(1)
                bt["ERROR_PCT"] = (
                    bt["ERROR_ABS"] / bt["y_real"].replace(0, float("nan")) * 100
                ).round(1)

                fig_bt = go.Figure()
                fig_bt.add_trace(go.Bar(
                    x=bt["ds"], y=bt["y_real"], name="Real",
                    marker_color=COLORS["primary"],
                ))
                fig_bt.add_trace(go.Scatter(
                    x=bt["ds"], y=bt["y_pred"], name="Predicho",
                    mode="lines+markers",
                    line=dict(color=COLORS["warning"], width=2, dash="dot"),
                    marker=dict(size=8),
                ))
                fig_bt.update_layout(
                    paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["background"],
                    font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"]),
                    margin=dict(l=40, r=20, t=30, b=40),
                    xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#E5E7EB", title="Toneladas"),
                    legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                    barmode="overlay", height=300,
                )
                st.plotly_chart(fig_bt, width="stretch")

# ============================================================
# TAB 2 — Por Proceso
# ============================================================
with tab2:
    seccion_titulo("Pronostico por Proceso", f"Selecciona un proceso — horizonte: {horizonte} meses")

    if df_proceso.empty:
        st.warning("Sin datos de serie por proceso. Verifica que ventas_limpias tenga columna PROCESO y FECHAEMB.")
    else:
        procesos_disp = sorted(df_proceso["PROCESO"].dropna().unique().tolist())
        proceso_sel = st.selectbox("Proceso:", options=procesos_disp, key="fc_proceso_sel")

        if proceso_sel:
            df_proc_sel = filtrar_por_dimension(
                df_proceso, col_dim="PROCESO", valor=proceso_sel,
                col_periodo="PERIODO", col_val="PESO_TON",
            )

            if df_proc_sel.empty or len(df_proc_sel) < 12:
                st.warning(f"Datos insuficientes para el proceso: {proceso_sel} ({len(df_proc_sel)} periodos). Se requieren al menos 12.")
            else:
                with st.spinner(f"Calculando pronostico para {proceso_sel}…"):
                    res_proc = generar_forecast(df_proc_sel, horizonte=horizonte)

                if res_proc.error_msg:
                    st.error(f"Error: {res_proc.error_msg}")
                else:
                    col_m, col_mt = st.columns([1, 3])
                    with col_m:
                        st.markdown(
                            f"""
                            <div style='background:{COLORS["surface"]};border:1px solid #E5E7EB;
                            border-left:4px solid {COLORS["secondary"]};border-radius:8px;padding:14px;'>
                                <div style='color:{COLORS["text_light"]};font-size:0.78rem;'>MODELO</div>
                                <div style='color:{COLORS["secondary"]};font-size:1.2rem;font-weight:700;'>{res_proc.modelo}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_mt:
                        if res_proc.metricas:
                            tabla_metricas(res_proc.metricas, titulo=f"Metricas — {proceso_sel}")

                    fig_proc = _grafico_forecast(
                        res_proc, titulo=f"Pronostico — Proceso: {proceso_sel}"
                    )
                    st.plotly_chart(fig_proc, width="stretch")

                    df_fut_proc = _tabla_forecast_futuro(res_proc)
                    if not df_fut_proc.empty:
                        seccion_titulo(f"Tabla pronostico — {proceso_sel}")
                        tabla_ejecutiva(
                            df_fut_proc,
                            col_formatos={"FORECAST_TON": "{:,.1f}", "LOWER_90": "{:,.1f}", "UPPER_90": "{:,.1f}"},
                            key="fc_proc_tabla",
                            height=260,
                        )

        # Resumen comparativo de todos los procesos
        st.divider()
        with st.expander("Ver resumen comparativo de todos los procesos (puede tardar)"):
            if st.button("Generar resumen de procesos", key="btn_resumen_proc"):
                resultados_proc = {}
                progress = st.progress(0)
                procs = procesos_disp
                for i, proc in enumerate(procs):
                    df_p = filtrar_por_dimension(
                        df_proceso, col_dim="PROCESO", valor=proc,
                        col_periodo="PERIODO", col_val="PESO_TON",
                    )
                    if len(df_p) >= 12:
                        r = generar_forecast(df_p, horizonte=horizonte)
                        if not r.error_msg:
                            fut = _tabla_forecast_futuro(r)
                            if not fut.empty:
                                resultados_proc[proc] = fut["FORECAST_TON"].sum()
                    progress.progress((i + 1) / len(procs))
                progress.empty()

                if resultados_proc:
                    df_res = pd.DataFrame(
                        list(resultados_proc.items()),
                        columns=["PROCESO", "FORECAST_TOTAL_TON"]
                    ).sort_values("FORECAST_TOTAL_TON", ascending=True)

                    fig_res = go.Figure(go.Bar(
                        x=df_res["FORECAST_TOTAL_TON"],
                        y=df_res["PROCESO"],
                        orientation="h",
                        marker_color=COLORS["secondary"],
                        text=df_res["FORECAST_TOTAL_TON"].round(0),
                        textposition="outside",
                        hovertemplate="<b>%{y}</b><br>Forecast total: %{x:,.1f} ton<extra></extra>",
                    ))
                    fig_res.update_layout(
                        paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["background"],
                        font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"]),
                        margin=dict(l=10, r=80, t=50, b=40),
                        xaxis=dict(title=f"Forecast acumulado {horizonte} meses (ton)", gridcolor="#E5E7EB"),
                        yaxis=dict(title=""),
                        title=dict(text=f"Demanda esperada por proceso — proximos {horizonte} meses",
                                   font=dict(size=13, color=COLORS["primary"]), x=0),
                        showlegend=False,
                        height=max(300, len(df_res) * 36 + 80),
                    )
                    st.plotly_chart(fig_res, width="stretch")

# ============================================================
# TAB 3 — Por Familia (Producto)
# ============================================================
with tab3:
    seccion_titulo("Pronostico por Familia de Producto", f"Selecciona un producto — horizonte: {horizonte} meses")

    if df_mensual.empty or "PRODUCTO_LIMPIO" not in df_mensual.columns:
        st.warning("Sin datos granulares por producto.")
    else:
        excluir_fc = {"OTROS", "OTHER", "N/D", "SIN CLASIFICAR", "S/C"}
        productos_disp = sorted([
            p for p in df_mensual["PRODUCTO_LIMPIO"].dropna().unique()
            if str(p).upper() not in excluir_fc
        ])
        prod_sel = st.selectbox("Producto:", options=productos_disp, key="fc_prod_sel")

        if prod_sel:
            df_prod_sel = filtrar_por_dimension(
                df_mensual, col_dim="PRODUCTO_LIMPIO", valor=prod_sel,
                col_periodo="PERIODO", col_val="PESO_TON",
            )

            if df_prod_sel.empty or len(df_prod_sel) < 12:
                st.warning(f"Datos insuficientes para {prod_sel} ({len(df_prod_sel)} periodos).")
            else:
                with st.spinner(f"Calculando pronostico para {prod_sel}…"):
                    res_prod = generar_forecast(df_prod_sel, horizonte=horizonte)

                if res_prod.error_msg:
                    st.error(f"Error: {res_prod.error_msg}")
                else:
                    col_m2, col_mt2 = st.columns([1, 3])
                    with col_m2:
                        st.markdown(
                            f"""
                            <div style='background:{COLORS["surface"]};border:1px solid #E5E7EB;
                            border-left:4px solid {COLORS["accent"]};border-radius:8px;padding:14px;'>
                                <div style='color:{COLORS["text_light"]};font-size:0.78rem;'>MODELO</div>
                                <div style='color:{COLORS["accent"]};font-size:1.2rem;font-weight:700;'>{res_prod.modelo}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_mt2:
                        if res_prod.metricas:
                            tabla_metricas(res_prod.metricas, titulo=f"Metricas — {prod_sel}")

                    fig_prod = _grafico_forecast(
                        res_prod, titulo=f"Pronostico — Familia: {prod_sel}"
                    )
                    st.plotly_chart(fig_prod, width="stretch")

                    df_fut_prod = _tabla_forecast_futuro(res_prod)
                    if not df_fut_prod.empty:
                        seccion_titulo(f"Tabla pronostico — {prod_sel}")
                        tabla_ejecutiva(
                            df_fut_prod,
                            col_formatos={"FORECAST_TON": "{:,.1f}", "LOWER_90": "{:,.1f}", "UPPER_90": "{:,.1f}"},
                            key="fc_prod_tabla",
                            height=260,
                        )

        # Resumen comparativo de todas las familias
        st.divider()
        with st.expander("Ver resumen comparativo de todas las familias (puede tardar)"):
            if st.button("Generar resumen de familias", key="btn_resumen_prod"):
                resultados_fam = {}
                progress2 = st.progress(0)
                for i, prod in enumerate(productos_disp):
                    df_p2 = filtrar_por_dimension(
                        df_mensual, col_dim="PRODUCTO_LIMPIO", valor=prod,
                        col_periodo="PERIODO", col_val="PESO_TON",
                    )
                    if len(df_p2) >= 12:
                        r2 = generar_forecast(df_p2, horizonte=horizonte)
                        if not r2.error_msg:
                            fut2 = _tabla_forecast_futuro(r2)
                            if not fut2.empty:
                                resultados_fam[prod] = fut2["FORECAST_TON"].sum()
                    progress2.progress((i + 1) / len(productos_disp))
                progress2.empty()

                if resultados_fam:
                    df_fam = pd.DataFrame(
                        list(resultados_fam.items()),
                        columns=["FAMILIA", "FORECAST_TOTAL_TON"]
                    ).sort_values("FORECAST_TOTAL_TON", ascending=True)

                    fig_fam = go.Figure(go.Bar(
                        x=df_fam["FORECAST_TOTAL_TON"],
                        y=df_fam["FAMILIA"],
                        orientation="h",
                        marker_color=COLORS["accent"],
                        text=df_fam["FORECAST_TOTAL_TON"].round(0),
                        textposition="outside",
                        hovertemplate="<b>%{y}</b><br>Forecast: %{x:,.1f} ton<extra></extra>",
                    ))
                    fig_fam.update_layout(
                        paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["background"],
                        font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"]),
                        margin=dict(l=10, r=80, t=50, b=40),
                        xaxis=dict(title=f"Forecast acumulado {horizonte} meses (ton)", gridcolor="#E5E7EB"),
                        yaxis=dict(title=""),
                        title=dict(text=f"Demanda esperada por familia — proximos {horizonte} meses",
                                   font=dict(size=13, color=COLORS["primary"]), x=0),
                        showlegend=False,
                        height=max(300, len(df_fam) * 36 + 80),
                    )
                    st.plotly_chart(fig_fam, width="stretch")
