"""
kpi_cards.py — Tarjetas KPI ejecutivas reutilizables.
Renderiza métricas con formato corporativo TYASA.
"""

import streamlit as st
from config import COLORS


def kpi_card(
    label: str,
    value: str | float | int,
    delta: str | float | None = None,
    delta_label: str = "",
    icon: str = "",
    suffix: str = "",
    prefix: str = "",
    help_text: str | None = None,
) -> None:
    """
    Renderiza una tarjeta KPI ejecutiva.

    Args:
        label: nombre del indicador.
        value: valor principal.
        delta: variación (positivo = verde, negativo = rojo).
        delta_label: texto adicional junto al delta.
        icon: emoji o ícono opcional.
        suffix: sufijo del valor (ej. " ton", " %").
        prefix: prefijo del valor (ej. "$").
        help_text: tooltip informativo.
    """
    # Formatear valor
    if isinstance(value, float):
        value_str = f"{prefix}{value:,.1f}{suffix}"
    elif isinstance(value, int):
        value_str = f"{prefix}{value:,}{suffix}"
    else:
        value_str = f"{prefix}{value}{suffix}"

    # Color y flecha del delta
    delta_html = ""
    if delta is not None:
        try:
            delta_num = float(delta)
            color = COLORS["success"] if delta_num >= 0 else COLORS["danger"]
            arrow = "▲" if delta_num >= 0 else "▼"
            delta_html = (
                f"<span style='color:{color};font-size:0.82rem;font-weight:600;'>"
                f"{arrow} {abs(delta_num):.1f}%"
                f"</span>"
                f"<span style='color:{COLORS['text_light']};font-size:0.78rem;'> {delta_label}</span>"
            )
        except (TypeError, ValueError):
            delta_html = f"<span style='color:{COLORS['neutral']};font-size:0.82rem;'>{delta}</span>"

    help_attr = f'title="{help_text}"' if help_text else ""

    st.markdown(
        f"""
        <div {help_attr} style='
            background:{COLORS["surface"]};
            border:1px solid #E5E7EB;
            border-left:4px solid {COLORS["primary"]};
            border-radius:8px;
            padding:16px 20px;
            margin-bottom:8px;
        '>
            <div style='
                color:{COLORS["text_light"]};
                font-size:0.78rem;
                font-weight:600;
                text-transform:uppercase;
                letter-spacing:0.05em;
                margin-bottom:4px;
            '>{icon} {label}</div>
            <div style='
                color:{COLORS["primary"]};
                font-size:1.8rem;
                font-weight:700;
                line-height:1.2;
                margin-bottom:4px;
            '>{value_str}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(kpis: list[dict]) -> None:
    """
    Renderiza una fila de tarjetas KPI en columnas iguales.

    Args:
        kpis: lista de dicts con claves: label, value, delta, delta_label, icon, suffix, prefix, help_text.
              Todas las claves excepto 'label' y 'value' son opcionales.

    Example:
        render_kpi_row([
            {"label": "Toneladas", "value": 12540, "suffix": " ton", "icon": "⚖️"},
            {"label": "Clientes", "value": 87, "icon": "👥"},
        ])
    """
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            kpi_card(
                label=kpi.get("label", ""),
                value=kpi.get("value", 0),
                delta=kpi.get("delta"),
                delta_label=kpi.get("delta_label", "vs mes anterior"),
                icon=kpi.get("icon", ""),
                suffix=kpi.get("suffix", ""),
                prefix=kpi.get("prefix", ""),
                help_text=kpi.get("help_text"),
            )


def seccion_titulo(titulo: str, subtitulo: str = "") -> None:
    """Renderiza un encabezado de sección con estilo corporativo."""
    st.markdown(
        f"""
        <div style='margin:24px 0 12px 0;'>
            <h3 style='color:{COLORS["primary"]};margin:0;font-size:1.15rem;'>
                {titulo}
            </h3>
            {"<p style='color:" + COLORS["text_light"] + ";font-size:0.85rem;margin:2px 0 0 0;'>" + subtitulo + "</p>" if subtitulo else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )
