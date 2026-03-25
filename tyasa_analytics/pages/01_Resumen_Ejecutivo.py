"""
01_Resumen_Ejecutivo.py — Visión 360° de la demanda histórica.
KPIs, tendencia mensual, participación por producto, top clientes y top productos.
"""

import streamlit as st
import pandas as pd
from config import APP_NAME, COLORS

from data.loaders import (
    load_gold_demanda_cliente,
    load_gold_demanda_producto,
    load_gold_demanda_mensual,
    load_gold_demanda_mensual_total,
)
from analytics.kpis import calcular_kpis_resumen, calcular_top_n, calcular_participacion
from analytics.series_tiempo import preparar_serie_mensual, construir_heatmap_mes_anio
from components.kpi_cards import render_kpi_row, seccion_titulo
from components.filters import (
    sidebar_header,
    filtro_rango_fechas,
    aplicar_filtro_fechas,
)
from components.charts import linea_temporal, barras_horizontales, donut, heatmap, treemap
from components.tables import tabla_ejecutiva

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"Resumen Ejecutivo — {APP_NAME}",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — filtros
# ---------------------------------------------------------------------------
sidebar_header("Filtros", "📊")
fecha_inicio, fecha_fin = filtro_rango_fechas(key_prefix="re")

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
with st.spinner("Cargando datos…"):
    df_cliente = load_gold_demanda_cliente()
    df_producto = load_gold_demanda_producto()
    df_mensual = load_gold_demanda_mensual_total()   # serie total por mes

# Aplicar filtro de fechas a la serie mensual
df_mensual_f = aplicar_filtro_fechas(df_mensual, fecha_inicio, fecha_fin, col="PERIODO")

# ---------------------------------------------------------------------------
# Encabezado de página
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <h2 style='color:{COLORS["primary"]};margin-bottom:0;'>📊 Resumen Ejecutivo</h2>
    <p style='color:{COLORS["text_light"]};'>Aceros Planos Negros — Demanda histórica</p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# KPIs principales
# ---------------------------------------------------------------------------
kpis_data = calcular_kpis_resumen(df_cliente, df_producto, df_mensual_f)

render_kpi_row([
    {
        "label": "Toneladas Totales",
        "value": kpis_data.toneladas_totales,
        "suffix": " ton",
        "icon": "⚖️",
        "help_text": "Suma total de toneladas en el período seleccionado.",
    },
    {
        "label": "Clientes Activos",
        "value": kpis_data.clientes_activos,
        "icon": "👥",
        "help_text": "Número de clientes con al menos un embarque.",
    },
    {
        "label": "Productos Activos",
        "value": kpis_data.productos_activos,
        "icon": "🔩",
        "help_text": "Número de productos únicos vendidos.",
    },
    {
        "label": "Ticket Prom. / Cliente",
        "value": kpis_data.ticket_promedio,
        "suffix": " ton",
        "icon": "📦",
        "delta": kpis_data.variacion_mom,
        "delta_label": "vs mes anterior",
        "help_text": "Toneladas promedio por cliente activo.",
    },
])

st.markdown("")

# Top cliente y producto
col1, col2 = st.columns(2)
with col1:
    st.metric("🥇 Top cliente", kpis_data.top_cliente or "—")
with col2:
    st.metric("🥇 Top producto", kpis_data.top_producto or "—")

st.divider()

# ---------------------------------------------------------------------------
# Sección 1 — Tendencia mensual
# ---------------------------------------------------------------------------
seccion_titulo("Tendencia Mensual de Demanda", "Serie de toneladas por mes")

if df_mensual_f.empty:
    st.warning("Sin datos para el período seleccionado.")
else:
    serie = preparar_serie_mensual(df_mensual_f)
    fig_linea = linea_temporal(
        serie,
        x="PERIODO",
        y="PESO_TON",
        titulo="Toneladas mensuales",
        y_label="Toneladas",
        show_area=True,
    )
    st.plotly_chart(fig_linea, use_container_width=True)

# ---------------------------------------------------------------------------
# Sección 2 — Participación por producto y Heatmap
# ---------------------------------------------------------------------------
col_a, col_b = st.columns([3, 2])

with col_a:
    seccion_titulo("Participación por Producto", "Distribución del volumen")
    df_part = calcular_participacion(df_producto, "PRODUCTO_LIMPIO")
    if not df_part.empty:
        fig_treemap = treemap(
            df_part,
            path=["PRODUCTO_LIMPIO"],
            values="PESO_TON",
            titulo="Treemap — Toneladas por producto",
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

with col_b:
    seccion_titulo("Mix Donut", "")
    if not df_part.empty:
        top_prod_donut = df_part.head(8)
        fig_donut = donut(top_prod_donut, names="PRODUCTO_LIMPIO", values="PESO_TON", titulo="")
        st.plotly_chart(fig_donut, use_container_width=True)

# ---------------------------------------------------------------------------
# Sección 3 — Heatmap Mes × Año
# ---------------------------------------------------------------------------
seccion_titulo("Heatmap Mes × Año", "Intensidad de demanda por período")

if not df_mensual_f.empty and "ANIO" in preparar_serie_mensual(df_mensual_f).columns:
    pivot = construir_heatmap_mes_anio(preparar_serie_mensual(df_mensual_f))
    if not pivot.empty:
        fig_heat = heatmap(pivot, titulo="Toneladas por Mes y Año", x_label="Año", y_label="Mes")
        st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------------------------
# Sección 4 — Top 10 Clientes y Top 10 Productos
# ---------------------------------------------------------------------------
col_c, col_d = st.columns(2)

with col_c:
    seccion_titulo("Top 10 Clientes", "Por volumen de toneladas")
    top_clientes = calcular_top_n(df_cliente, "CLIENTE", n=10)
    if not top_clientes.empty:
        fig_cli = barras_horizontales(
            top_clientes,
            x="PESO_TON",
            y="CLIENTE",
            titulo="",
            x_label="Toneladas",
        )
        st.plotly_chart(fig_cli, use_container_width=True)

with col_d:
    seccion_titulo("Top 10 Productos", "Por volumen de toneladas")
    top_prod = calcular_top_n(df_producto, "PRODUCTO_LIMPIO", n=10)
    if not top_prod.empty:
        fig_prod = barras_horizontales(
            top_prod,
            x="PESO_TON",
            y="PRODUCTO_LIMPIO",
            titulo="",
            x_label="Toneladas",
        )
        st.plotly_chart(fig_prod, use_container_width=True)

# ---------------------------------------------------------------------------
# Sección 5 — Tabla resumen exportable
# ---------------------------------------------------------------------------
st.divider()
seccion_titulo("Tabla Resumen por Cliente", "Descargable en Excel")

df_resumen = df_cliente.copy() if not df_cliente.empty else pd.DataFrame()
if not df_resumen.empty:
    tabla_ejecutiva(
        df_resumen.sort_values("PESO_TON", ascending=False),
        titulo="",
        col_formatos={"PESO_TON": "{:,.1f}"},
        key="resumen_clientes",
    )
