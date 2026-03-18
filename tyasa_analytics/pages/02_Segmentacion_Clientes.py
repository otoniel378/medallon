"""
02_Segmentacion_Clientes.py — Clasificacion ABC, Pareto y diversificacion.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import APP_NAME, COLORS, COLOR_SEQUENCE

from data.loaders import (
    load_gold_demanda_cliente,
    load_gold_cliente_producto,
    load_ventas_limpias,
)
from analytics.segmentacion import (
    clasificar_abc,
    resumen_abc,
    calcular_diversificacion,
    clientes_monoproducto,
)
from components.kpi_cards import render_kpi_row, seccion_titulo
from components.filters import sidebar_header, filtro_clientes, aplicar_filtro_lista
from components.charts import pareto, barras_horizontales, donut
from components.tables import tabla_ejecutiva, tabla_clasificacion_abc

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"Segmentacion — {APP_NAME}",
    page_icon="👥",
    layout="wide",
)

sidebar_header("Filtros", "👥")
clientes_sel = filtro_clientes(key_prefix="seg")

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <h2 style='color:{COLORS["primary"]};margin-bottom:0;'>👥 Segmentacion de Clientes</h2>
    <p style='color:{COLORS["text_light"]};'>Analisis Pareto, clasificacion ABC y diversificacion de portafolio</p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Filtro de año — INLINE, al inicio de la página
# ---------------------------------------------------------------------------
with st.spinner("Cargando datos base…"):
    df_vl_all = load_ventas_limpias()

anios_disp = []
if not df_vl_all.empty and "FECHAEMB" in df_vl_all.columns:
    anios_disp = sorted(df_vl_all["FECHAEMB"].dropna().dt.year.unique().tolist(), reverse=True)

col_flt1, col_flt2, col_flt3 = st.columns([1, 1, 4])
with col_flt1:
    anio_sel = st.selectbox(
        "Filtrar por año",
        options=["Todos"] + [str(a) for a in anios_disp],
        key="seg_anio",
    )

# ---------------------------------------------------------------------------
# Carga de datos según filtro de año
# ---------------------------------------------------------------------------
with st.spinner("Procesando datos…"):
    if anio_sel != "Todos" and not df_vl_all.empty:
        df_yr = df_vl_all[df_vl_all["FECHAEMB"].dt.year == int(anio_sel)].copy()
        df_cliente = (
            df_yr.groupby("CLIENTE", as_index=False)["PESO_TON"].sum()
            if "CLIENTE" in df_yr.columns else load_gold_demanda_cliente()
        )
        if "PRODUCTO_LIMPIO" in df_yr.columns and "CLIENTE" in df_yr.columns:
            df_cp = df_yr.groupby(["CLIENTE", "PRODUCTO_LIMPIO"], as_index=False)["PESO_TON"].sum()
        else:
            df_cp = load_gold_cliente_producto()
    else:
        df_cliente = load_gold_demanda_cliente()
        df_cp = load_gold_cliente_producto()

if clientes_sel:
    df_cliente = aplicar_filtro_lista(df_cliente, clientes_sel, "CLIENTE")
    df_cp = aplicar_filtro_lista(df_cp, clientes_sel, "CLIENTE")

label_yr = f" — Año {anio_sel}" if anio_sel != "Todos" else " — Histórico total"
st.caption(f"Mostrando datos: **{label_yr.strip(' — ')}**  ·  {len(df_cliente):,} clientes")
st.divider()

# ---------------------------------------------------------------------------
# Clasificacion ABC — KPIs
# ---------------------------------------------------------------------------
df_abc = clasificar_abc(df_cliente)
df_resumen_abc = resumen_abc(df_abc)

total_clientes = len(df_abc)
n_a = len(df_abc[df_abc["CLASE"] == "A"]) if not df_abc.empty else 0
n_b = len(df_abc[df_abc["CLASE"] == "B"]) if not df_abc.empty else 0
n_c = len(df_abc[df_abc["CLASE"] == "C"]) if not df_abc.empty else 0
pct_a = round(n_a / total_clientes * 100, 1) if total_clientes > 0 else 0

render_kpi_row([
    {"label": "Total Clientes", "value": total_clientes, "icon": "👥"},
    {"label": "Clase A (80% vol.)", "value": n_a,
     "suffix": f" ({pct_a}%)", "icon": "🥇",
     "help_text": "Clientes que representan el 80% del volumen."},
    {"label": "Clase B", "value": n_b, "icon": "🥈"},
    {"label": "Clase C", "value": n_c, "icon": "🥉"},
])
st.divider()

# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------
seccion_titulo("Analisis Pareto de Clientes", "Volumen acumulado — top clientes")

# Filtro inline: cantidad de clientes a mostrar
col_p1, col_p2, _ = st.columns([1, 1, 4])
with col_p1:
    top_n_pareto = st.selectbox("Top clientes", options=[10, 15, 20, 30], index=2, key="pareto_top")

if not df_abc.empty:
    df_par = df_abc.head(top_n_pareto).copy()
    df_par["CLIENTE_CORTO"] = df_par["CLIENTE"].str[:26]
    fig_pareto = pareto(
        df_par, x="CLIENTE_CORTO", y="PESO_TON",
        titulo=f"Pareto — Top {top_n_pareto} Clientes por Toneladas", max_items=top_n_pareto,
    )
    fig_pareto.update_layout(
        height=440,
        margin=dict(b=140, l=40, r=80, t=50),
        xaxis=dict(tickfont=dict(size=8), tickangle=-50),
    )
    st.plotly_chart(fig_pareto, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Resumen ABC + filtro interactivo de tabla
# ---------------------------------------------------------------------------
seccion_titulo("Clasificacion ABC", "Selecciona clase para filtrar la tabla de clientes")

col_a, col_b = st.columns([1, 2])

with col_a:
    if not df_resumen_abc.empty:
        fig_donut = donut(df_resumen_abc, names="CLASE", values="PESO_TON", titulo="Volumen por clase")
        st.plotly_chart(fig_donut, width="stretch")
        tabla_ejecutiva(
            df_resumen_abc,
            col_formatos={"PESO_TON": "{:,.1f}", "PCT_VOLUMEN": "{:.1f}%"},
            key="resumen_abc",
            height=140,
        )

with col_b:
    clase_sel = st.radio(
        "Mostrar clientes de clase:",
        options=["Todas", "A", "B", "C"],
        horizontal=True,
        key="abc_clase_radio",
    )
    if not df_abc.empty:
        df_det = df_abc if clase_sel == "Todas" else df_abc[df_abc["CLASE"] == clase_sel]
        st.caption(f"{len(df_det):,} clientes en clase **{clase_sel}**")
        cols_ok = [c for c in ["RANK", "CLIENTE", "PESO_TON", "PCT", "PCT_ACUM", "CLASE"] if c in df_det.columns]
        tabla_clasificacion_abc(df_det[cols_ok], key="abc_detalle")

st.divider()

# ---------------------------------------------------------------------------
# Diversificacion
# ---------------------------------------------------------------------------
seccion_titulo("Diversificacion por Cliente", "Numero de productos distintos comprados")

df_div = calcular_diversificacion(df_cp)

if not df_div.empty:
    col_c, col_d = st.columns(2)

    with col_c:
        # Filtro inline: top N clientes
        top_div = st.selectbox("Top clientes", [10, 15, 20], index=2, key="div_top")
        fig_div = barras_horizontales(
            df_div.head(top_div),
            x="N_PRODUCTOS", y="CLIENTE",
            titulo=f"Top {top_div} clientes mas diversificados",
            x_label="N de productos",
        )
        st.plotly_chart(fig_div, width="stretch")

    with col_d:
        seccion_titulo("Clientes Monoproducto", "Solo compran un tipo")
        df_mono = clientes_monoproducto(df_cp)
        tabla_ejecutiva(
            df_mono,
            col_formatos={"PESO_TON": "{:,.1f}"},
            key="mono_producto",
            height=360,
        )

# ---------------------------------------------------------------------------
# Matriz Cliente x Producto — Barras Apiladas
# ---------------------------------------------------------------------------
st.divider()
seccion_titulo("Mix de Productos por Cliente", "Distribucion de toneladas — barras apiladas")

if not df_cp.empty and "PRODUCTO_LIMPIO" in df_cp.columns:
    # Filtro inline: número de clientes
    col_m1, col_m2, _ = st.columns([1, 1, 4])
    with col_m1:
        top_cli_mat = st.selectbox("Top clientes", [10, 15, 20, 30], index=2, key="mat_top")

    top_ids = (
        df_cp.groupby("CLIENTE")["PESO_TON"].sum()
        .nlargest(top_cli_mat).index.tolist()
    )
    df_bar = df_cp[df_cp["CLIENTE"].isin(top_ids)].copy()
    df_bar["CLIENTE_CORTO"] = df_bar["CLIENTE"].str[:28]

    orden_cli = (
        df_bar.groupby("CLIENTE_CORTO")["PESO_TON"].sum()
        .sort_values(ascending=True).index.tolist()
    )

    fig_stack = px.bar(
        df_bar,
        x="PESO_TON",
        y="CLIENTE_CORTO",
        color="PRODUCTO_LIMPIO",
        orientation="h",
        title=f"Toneladas por cliente y producto (top {top_cli_mat})",
        labels={"PESO_TON": "Toneladas", "CLIENTE_CORTO": "", "PRODUCTO_LIMPIO": "Producto"},
        color_discrete_sequence=COLOR_SEQUENCE,
        category_orders={"CLIENTE_CORTO": orden_cli},
    )
    fig_stack.update_layout(
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"], size=10),
        margin=dict(l=10, r=20, t=50, b=40),
        legend=dict(
            orientation="h", y=-0.18, x=0.5, xanchor="center",
            font=dict(size=9), title_text="",
        ),
        height=max(380, top_cli_mat * 28 + 100),
        barmode="stack",
        xaxis=dict(title="Toneladas", gridcolor="#E5E7EB"),
        yaxis=dict(title="", tickfont=dict(size=9)),
        title=dict(font=dict(size=14, color=COLORS["primary"]), x=0),
    )
    st.plotly_chart(fig_stack, width="stretch")
