"""
app.py — Punto de entrada principal de tyasa_analytics.
Configura la página, carga estilos y muestra landing de bienvenida.
"""

import os
import streamlit as st
from config import APP_NAME, APP_SUBTITLE, APP_ICON, ASSETS_DIR, COLORS

# ---------------------------------------------------------------------------
# Configuración de página (debe ser el primer comando Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": f"**{APP_NAME}** — Plataforma de inteligencia comercial para TYASA.",
    },
)

# ---------------------------------------------------------------------------
# Carga de estilos CSS personalizados
# ---------------------------------------------------------------------------
css_path = os.path.join(ASSETS_DIR, "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 5])

with col_logo:
    logo_path = os.path.join(ASSETS_DIR, "logo_tyasa.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    else:
        st.markdown(
            f"<div style='font-size:48px;text-align:center;'>🔩</div>",
            unsafe_allow_html=True,
        )

with col_title:
    st.markdown(
        f"""
        <h1 style='color:{COLORS["primary"]};margin-bottom:0;'>{APP_NAME}</h1>
        <p style='color:{COLORS["neutral"]};font-size:1.1rem;margin-top:4px;'>
            {APP_SUBTITLE}
        </p>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Descripción y acceso a módulos
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <h3 style='color:{COLORS["primary"]}'>Plataforma de Inteligencia Comercial</h3>
    <p style='color:{COLORS["text"]};'>
        Selecciona un módulo en el menú lateral para comenzar el análisis.
    </p>
    """,
    unsafe_allow_html=True,
)

modules = [
    ("📊", "Resumen Ejecutivo", "Visión 360° de la demanda histórica: KPIs, tendencias, participación de producto y top clientes."),
    ("👥", "Segmentación de Clientes", "Análisis Pareto, clasificación ABC, concentración de demanda y diversificación."),
    ("📈", "Series de Tiempo", "Comportamiento temporal de la demanda, variaciones mensuales, volatilidad y estabilidad."),
    ("🔮", "Forecasting", "Pronósticos de demanda mensual con Prophet/ARIMA, backtesting y métricas de error."),
    ("🎯", "Mix de Productos", "Análisis de portafolio, co-ocurrencia y oportunidades básicas de cross-sell."),
]

cols = st.columns(len(modules))
for col, (icon, title, desc) in zip(cols, modules):
    with col:
        st.markdown(
            f"""
            <div style='
                background:{COLORS["surface"]};
                border:1px solid #E5E7EB;
                border-top:3px solid {COLORS["primary"]};
                border-radius:8px;
                padding:20px 16px;
                min-height:180px;
                text-align:center;
            '>
                <div style='font-size:2rem;'>{icon}</div>
                <h4 style='color:{COLORS["primary"]};margin:8px 0;font-size:0.95rem;'>{title}</h4>
                <p style='color:{COLORS["text_light"]};font-size:0.82rem;'>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.caption(f"© 2024 TYASA — {APP_NAME} v1.0 | Datos: Aceros Planos Negros")
