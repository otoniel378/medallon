"""
03_Series_de_Tiempo.py — Analisis temporal de la demanda.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from config import APP_NAME, COLORS, COLOR_SEQUENCE

from data.loaders import (
    load_gold_demanda_mensual,
    load_gold_demanda_mensual_total,
    load_serie_mensual_proceso,
    load_serie_mensual_cliente,
)
from analytics.series_tiempo import (
    preparar_serie_mensual,
    calcular_variacion_mensual,
    calcular_volatilidad,
    top_afectados_variacion,
)
from components.kpi_cards import render_kpi_row, seccion_titulo
from components.filters import sidebar_header
from components.charts import linea_temporal
from components.tables import tabla_ejecutiva

st.set_page_config(
    page_title=f"Series de Tiempo — {APP_NAME}",
    page_icon="📈",
    layout="wide",
)

sidebar_header("Series de Tiempo", "📈")

# ---------------------------------------------------------------------------
# Carga de datos (sin filtro inicial — el usuario filtra inline por sección)
# ---------------------------------------------------------------------------
with st.spinner("Cargando datos…"):
    df_mensual_total = load_gold_demanda_mensual_total()
    df_mensual_gran  = load_gold_demanda_mensual()
    df_proceso_ts    = load_serie_mensual_proceso()
    df_cliente_ts    = load_serie_mensual_cliente()

# Preparar listas de años disponibles
def _anios_de(df):
    if df.empty or "PERIODO" not in df.columns:
        return []
    col = pd.to_datetime(df["PERIODO"], errors="coerce")
    return sorted(col.dropna().dt.year.unique().tolist(), reverse=True)

anios_total = _anios_de(df_mensual_total)

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <h2 style='color:{COLORS["primary"]};margin-bottom:0;'>📈 Series de Tiempo</h2>
    <p style='color:{COLORS["text_light"]};'>Comportamiento temporal de la demanda de Aceros Planos Negros</p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ============================================================
# SECCION 1 — Serie mensual total
# ============================================================
seccion_titulo("Serie Mensual Total", "Toneladas acumuladas por mes")

# Filtro inline: rango de años
col_s1, col_s2, _ = st.columns([1, 1, 4])
with col_s1:
    anio_ini_sel = st.selectbox(
        "Desde año",
        options=["Todos"] + [str(a) for a in anios_total],
        key="st_anio_ini",
    )
with col_s2:
    anio_fin_sel = st.selectbox(
        "Hasta año",
        options=["Todos"] + [str(a) for a in anios_total],
        index=0,
        key="st_anio_fin",
    )

# Aplicar filtro
df_tot_f = df_mensual_total.copy()
if "PERIODO" in df_tot_f.columns:
    df_tot_f["PERIODO"] = pd.to_datetime(df_tot_f["PERIODO"], errors="coerce")
    if anio_ini_sel != "Todos":
        df_tot_f = df_tot_f[df_tot_f["PERIODO"].dt.year >= int(anio_ini_sel)]
    if anio_fin_sel != "Todos":
        df_tot_f = df_tot_f[df_tot_f["PERIODO"].dt.year <= int(anio_fin_sel)]

serie = preparar_serie_mensual(df_tot_f)
serie_var = calcular_variacion_mensual(serie)

if not serie_var.empty:
    ult = serie_var.iloc[-1]
    render_kpi_row([
        {"label": "Ultimo mes", "value": ult["PESO_TON"], "suffix": " ton", "icon": "📅",
         "delta": ult.get("VAR_MOM_PCT"), "delta_label": "MoM"},
        {"label": "Promedio mensual", "value": round(serie_var["PESO_TON"].mean(), 1),
         "suffix": " ton", "icon": "📊"},
        {"label": "Maximo", "value": round(serie_var["PESO_TON"].max(), 1), "suffix": " ton", "icon": "📈"},
        {"label": "Minimo", "value": round(serie_var["PESO_TON"].min(), 1), "suffix": " ton", "icon": "📉"},
    ])

    fig_serie = linea_temporal(
        serie_var, x="PERIODO", y="PESO_TON",
        titulo="Demanda mensual total", y_label="Toneladas", show_area=True,
    )
    st.plotly_chart(fig_serie, width="stretch")
else:
    st.warning("Sin datos para el periodo seleccionado.")

# ============================================================
# SECCION 2 — Variacion mensual + Top afectados
# ============================================================
st.divider()
seccion_titulo("Variacion Mensual (%)", "Cambio porcentual respecto al mes anterior")

if not serie_var.empty and "VAR_MOM_PCT" in serie_var.columns:
    df_var = serie_var.dropna(subset=["VAR_MOM_PCT"]).copy()
    df_var["COLOR"] = df_var["VAR_MOM_PCT"].apply(
        lambda v: COLORS["success"] if v >= 0 else COLORS["danger"]
    )
    fig_var = go.Figure()
    fig_var.add_trace(go.Bar(
        x=df_var["PERIODO"], y=df_var["VAR_MOM_PCT"],
        marker_color=df_var["COLOR"],
        hovertemplate="%{x|%b %Y}<br>Variacion: %{y:.1f}%<extra></extra>",
    ))
    fig_var.add_hline(y=0, line_color=COLORS["neutral"], line_width=1)
    fig_var.update_layout(
        paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"]),
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(showgrid=False), yaxis=dict(title="Variacion (%)", gridcolor="#E5E7EB"),
        showlegend=False, height=280,
    )
    st.plotly_chart(fig_var, width="stretch")

# Top 10 más afectados (usando el mismo rango de años filtrado)
st.markdown(
    f"<p style='color:{COLORS['text_light']};font-size:0.88rem;margin-top:4px;'>"
    "Top 10 dimensiones con mayor variacion absoluta en el ultimo periodo disponible</p>",
    unsafe_allow_html=True,
)

# Filtrar series granulares al mismo rango de años seleccionado
def _filtrar_anios(df, col="PERIODO"):
    if df.empty or col not in df.columns:
        return df
    d = df.copy()
    d[col] = pd.to_datetime(d[col], errors="coerce")
    if anio_ini_sel != "Todos":
        d = d[d[col].dt.year >= int(anio_ini_sel)]
    if anio_fin_sel != "Todos":
        d = d[d[col].dt.year <= int(anio_fin_sel)]
    return d

df_gran_f   = _filtrar_anios(df_mensual_gran)
df_proc_f   = _filtrar_anios(df_proceso_ts)
df_cli_f    = _filtrar_anios(df_cliente_ts)

col_p, col_proc, col_cli = st.columns(3)

def _render_top(df, col_dim, titulo, key_suffix):
    """Muestra tabla de top afectados por variacion."""
    if df.empty or col_dim not in df.columns:
        st.info("Sin datos.")
        return
    top = top_afectados_variacion(df, col_dim)
    if top.empty:
        st.info("Sin datos suficientes.")
        return
    top["DIMENSION"] = top["DIMENSION"].astype(str).str[:30]
    top["VAR_ABS"] = top["VAR_ABS"].round(1)
    top["SIGNO"] = top["VAR_ABS"].apply(lambda v: "▲" if v >= 0 else "▼")
    top["VAR_PCT_STR"] = top["VAR_PCT"].apply(
        lambda v: f"+{v:.1f}%" if pd.notna(v) and v > 0 else (f"{v:.1f}%" if pd.notna(v) else "—")
    )
    st.dataframe(
        top[["DIMENSION", "VAR_ABS", "VAR_PCT_STR"]].rename(columns={
            "DIMENSION": titulo, "VAR_ABS": "Var (ton)", "VAR_PCT_STR": "Var %"
        }),
        hide_index=True, use_container_width=True, height=310,
    )

with col_p:
    st.markdown(f"**📦 Productos**")
    _render_top(df_gran_f, "PRODUCTO_LIMPIO", "Producto", "prod")

with col_proc:
    st.markdown(f"**⚙️ Procesos**")
    _render_top(df_proc_f, "PROCESO", "Proceso", "proc")

with col_cli:
    st.markdown(f"**👥 Clientes**")
    _render_top(df_cli_f, "CLIENTE", "Cliente", "cli")

# ============================================================
# SECCION 3 — Tendencia mensual por producto
# ============================================================
st.divider()
seccion_titulo("Tendencia Mensual por Producto", "Evolucion historica por tipo de producto")

# Filtros inline
excluir_prods = {"OTROS", "OTHER", "N/D", "SIN CLASIFICAR", "S/C"}

df_gran_clean = df_gran_f.copy()
if "PRODUCTO_LIMPIO" in df_gran_clean.columns:
    df_gran_clean = df_gran_clean[
        ~df_gran_clean["PRODUCTO_LIMPIO"].str.upper().isin(excluir_prods)
    ]

prods_disponibles = sorted(df_gran_clean["PRODUCTO_LIMPIO"].dropna().unique().tolist()) if not df_gran_clean.empty else []

col_t1, col_t2, _ = st.columns([2, 1, 3])
with col_t1:
    prods_sel = st.multiselect(
        "Productos a mostrar",
        options=prods_disponibles,
        default=prods_disponibles[:min(8, len(prods_disponibles))],
        key="tend_prods",
    )
with col_t2:
    tipo_linea = st.selectbox("Tipo", ["Líneas", "Area apilada"], key="tend_tipo")

if not df_gran_clean.empty and prods_sel:
    df_tend = df_gran_clean[df_gran_clean["PRODUCTO_LIMPIO"].isin(prods_sel)].copy()

    if tipo_linea == "Area apilada":
        fig_tend = px.area(
            df_tend,
            x="PERIODO", y="PESO_TON",
            color="PRODUCTO_LIMPIO",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"PESO_TON": "Toneladas", "PERIODO": "", "PRODUCTO_LIMPIO": "Producto"},
            title="Tendencia mensual por producto (area apilada)",
        )
    else:
        fig_tend = px.line(
            df_tend,
            x="PERIODO", y="PESO_TON",
            color="PRODUCTO_LIMPIO",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"PESO_TON": "Toneladas", "PERIODO": "", "PRODUCTO_LIMPIO": "Producto"},
            title="Tendencia mensual por producto",
        )
        fig_tend.update_traces(line_width=2)

    fig_tend.update_layout(
        paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"], size=11),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            orientation="h", y=-0.22, x=0.5, xanchor="center",
            font=dict(size=10), title_text="",
        ),
        height=420,
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#E5E7EB", title="Toneladas"),
        title=dict(font=dict(size=14, color=COLORS["primary"]), x=0),
    )
    st.plotly_chart(fig_tend, width="stretch")
elif not prods_sel:
    st.info("Selecciona al menos un producto en el filtro de arriba.")

# ============================================================
# SECCION 4 — Volatilidad y ranking de estabilidad
# ============================================================
st.divider()
seccion_titulo("Variabilidad por Producto", "Coeficiente de variacion mensual (CV menor = mas estable)")

if not df_gran_clean.empty and "PRODUCTO_LIMPIO" in df_gran_clean.columns:
    # Filtro inline: productos a incluir
    col_v1, _ = st.columns([2, 4])
    with col_v1:
        prods_vol_sel = st.multiselect(
            "Incluir productos",
            options=prods_disponibles,
            default=prods_disponibles,
            key="vol_prods",
        )

    df_vol_input = df_gran_clean.copy()
    if prods_vol_sel:
        df_vol_input = df_vol_input[df_vol_input["PRODUCTO_LIMPIO"].isin(prods_vol_sel)]

    df_vol = calcular_volatilidad(df_vol_input, col_dim="PRODUCTO_LIMPIO", col_val="PESO_TON")

    if not df_vol.empty:
        color_map = {
            "Alta":  COLORS["success"],
            "Media": COLORS["warning"],
            "Baja":  COLORS["danger"],
        }
        df_vp = df_vol.sort_values("CV", ascending=True).copy()
        df_vp["COLOR"] = df_vp["ESTABILIDAD"].map(color_map)

        fig_vol = go.Figure()
        for est, color in color_map.items():
            sub = df_vp[df_vp["ESTABILIDAD"] == est]
            if sub.empty:
                continue
            fig_vol.add_trace(go.Bar(
                x=sub["CV"],
                y=sub["DIMENSION"],
                orientation="h",
                name=f"Estabilidad {est}",
                marker_color=color,
                text=sub["CV"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
                textfont=dict(size=9),
                hovertemplate="<b>%{y}</b><br>CV: %{x:.1f}%<extra></extra>",
            ))

        fig_vol.update_layout(
            paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["background"],
            font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"], size=10),
            margin=dict(l=10, r=90, t=50, b=40),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            height=max(320, len(df_vp) * 34 + 80),
            barmode="overlay",
            xaxis=dict(title="CV (%)", gridcolor="#E5E7EB"),
            yaxis=dict(tickfont=dict(size=9)),
            title=dict(
                text="Ranking de estabilidad — menor CV es más estable",
                font=dict(size=13, color=COLORS["primary"]), x=0,
            ),
        )
        st.plotly_chart(fig_vol, width="stretch")

        st.caption("CV calculado sobre la serie mensual historica de cada producto.")
        tabla_ejecutiva(
            df_vol[["DIMENSION", "MEDIA", "STD", "CV", "ESTABILIDAD"]].rename(
                columns={"DIMENSION": "PRODUCTO", "MEDIA": "PROM_TON", "STD": "DESV_STD"}
            ),
            col_formatos={"PROM_TON": "{:,.1f}", "DESV_STD": "{:,.1f}", "CV": "{:.1f}%"},
            key="estabilidad_prod",
            height=300,
        )

# ============================================================
# Tabla exportable
# ============================================================
st.divider()
seccion_titulo("Serie Historica Completa", "Exportable en Excel")

if not serie_var.empty:
    cols_exp = [c for c in ["PERIODO", "ANIO", "MES", "PESO_TON", "VAR_MOM", "VAR_MOM_PCT"] if c in serie_var.columns]
    tabla_ejecutiva(
        serie_var[cols_exp],
        col_formatos={"PESO_TON": "{:,.1f}", "VAR_MOM": "{:,.1f}", "VAR_MOM_PCT": "{:.1f}%"},
        key="serie_historica",
        height=380,
    )
