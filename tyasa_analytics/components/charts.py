"""
charts.py — Funciones reutilizables de gráficos con Plotly.
Todas retornan figuras de Plotly configuradas con la paleta TYASA.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from config import COLORS, COLOR_SEQUENCE, HEATMAP_COLORSCALE


_LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["surface"],
    plot_bgcolor=COLORS["background"],
    font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"], size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        xanchor="center",
        x=0.5,
        font=dict(size=11),
    ),
)


def linea_temporal(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    titulo: str = "",
    y_label: str = "Toneladas",
    show_area: bool = False,
) -> go.Figure:
    """
    Gráfico de línea temporal.

    Args:
        df: DataFrame con columnas x, y (y opcionalmente color).
        x: columna del eje X (fechas).
        y: columna del eje Y (valores).
        color: columna de segmentación opcional.
        titulo: título del gráfico.
        y_label: etiqueta del eje Y.
        show_area: si True, rellena área bajo la línea.
    """
    if df.empty:
        return _empty_fig(titulo)

    if color and color in df.columns:
        fig = px.line(
            df, x=x, y=y, color=color,
            color_discrete_sequence=COLOR_SEQUENCE,
            title=titulo,
        )
    else:
        fig = px.area(df, x=x, y=y, title=titulo) if show_area else px.line(df, x=x, y=y, title=titulo)
        fig.update_traces(line_color=COLORS["primary"])
        if show_area:
            fig.update_traces(fillcolor=f"rgba(27,58,92,0.15)")

    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis_title="",
        yaxis_title=y_label,
        title=dict(font=dict(size=14, color=COLORS["primary"]), x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E5E7EB")
    return fig


def barras_horizontales(
    df: pd.DataFrame,
    x: str,
    y: str,
    titulo: str = "",
    x_label: str = "Toneladas",
    max_items: int = 15,
) -> go.Figure:
    """
    Gráfico de barras horizontales ordenado descendente.
    """
    if df.empty:
        return _empty_fig(titulo)

    df_plot = df.nlargest(max_items, x).sort_values(x, ascending=True)
    fig = px.bar(
        df_plot,
        x=x,
        y=y,
        orientation="h",
        title=titulo,
        color_discrete_sequence=[COLORS["primary"]],
        text=x,
    )
    fig.update_traces(texttemplate="%{text:,.1f}", textposition="outside")
    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis_title=x_label,
        yaxis_title="",
        title=dict(font=dict(size=14, color=COLORS["primary"]), x=0),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB")
    fig.update_yaxes(showgrid=False)
    return fig


def donut(
    df: pd.DataFrame,
    names: str,
    values: str,
    titulo: str = "",
    hole: float = 0.55,
) -> go.Figure:
    """
    Gráfico de dona (donut) para participación porcentual.
    """
    if df.empty:
        return _empty_fig(titulo)

    fig = px.pie(
        df,
        names=names,
        values=values,
        hole=hole,
        title=titulo,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(
        textposition="outside",
        textinfo="percent+label",
        hovertemplate="%{label}<br>%{value:,.1f} ton<br>%{percent}",
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        showlegend=False,
        title=dict(font=dict(size=14, color=COLORS["primary"]), x=0),
    )
    return fig


def treemap(
    df: pd.DataFrame,
    path: list[str],
    values: str,
    titulo: str = "",
) -> go.Figure:
    """
    Treemap para visualización jerárquica de participación.
    """
    if df.empty:
        return _empty_fig(titulo)

    fig = px.treemap(
        df,
        path=path,
        values=values,
        title=titulo,
        color=values,
        color_continuous_scale=HEATMAP_COLORSCALE,
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(font=dict(size=14, color=COLORS["primary"]), x=0),
    )
    return fig


def heatmap(
    pivot_df: pd.DataFrame,
    titulo: str = "",
    x_label: str = "Año",
    y_label: str = "Mes",
    fmt: str = ".0f",
) -> go.Figure:
    """
    Heatmap a partir de un DataFrame pivotado.

    Args:
        pivot_df: DataFrame con index=eje Y, columns=eje X.
    """
    if pivot_df.empty:
        return _empty_fig(titulo)

    fig = go.Figure(
        go.Heatmap(
            z=pivot_df.values,
            x=[str(c) for c in pivot_df.columns],
            y=pivot_df.index.tolist(),
            colorscale=HEATMAP_COLORSCALE,
            hoverongaps=False,
            hovertemplate=f"<b>%{{y}} — %{{x}}</b><br>Toneladas: %{{z:{fmt}}}<extra></extra>",
            showscale=True,
        )
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=titulo, font=dict(size=14, color=COLORS["primary"]), x=0),
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return fig


def pareto(
    df: pd.DataFrame,
    x: str,
    y: str,
    y_acum: str = "PCT_ACUM",
    titulo: str = "Análisis Pareto",
    max_items: int = 30,
) -> go.Figure:
    """
    Gráfico de Pareto: barras de volumen + línea de % acumulado.

    Args:
        df: DataFrame con x, y (volumen) y y_acum (% acumulado).
        max_items: máximo de ítems a mostrar.
    """
    if df.empty:
        return _empty_fig(titulo)

    df_plot = df.head(max_items).copy()

    fig = go.Figure()

    # Barras de volumen
    fig.add_trace(
        go.Bar(
            x=df_plot[x],
            y=df_plot[y],
            name="Toneladas",
            marker_color=COLORS["primary"],
            hovertemplate=f"%{{x}}<br>Toneladas: %{{y:,.1f}}<extra></extra>",
        )
    )

    # Línea acumulada
    if y_acum in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot[x],
                y=df_plot[y_acum],
                name="% Acumulado",
                yaxis="y2",
                line=dict(color=COLORS["warning"], width=2.5, dash="dot"),
                mode="lines+markers",
                marker=dict(size=5),
                hovertemplate=f"%{{x}}<br>Acumulado: %{{y:.1f}}%<extra></extra>",
            )
        )
        # Línea de referencia 80%
        fig.add_hline(
            y=80,
            yref="y2",
            line=dict(color=COLORS["danger"], width=1.5, dash="dash"),
            annotation_text="80%",
            annotation_position="right",
        )

    # _LAYOUT_BASE ya tiene 'legend'; lo excluimos para sobreescribir manualmente
    layout = {k: v for k, v in _LAYOUT_BASE.items() if k != "legend"}
    fig.update_layout(
        **layout,
        title=dict(text=titulo, font=dict(size=14, color=COLORS["primary"]), x=0),
        xaxis=dict(title="", tickangle=-40, showgrid=False),
        yaxis=dict(title="Toneladas", gridcolor="#E5E7EB"),
        yaxis2=dict(
            title="% Acumulado",
            overlaying="y",
            side="right",
            range=[0, 105],
            showgrid=False,
            ticksuffix="%",
        ),
        legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"),
        barmode="overlay",
    )
    return fig


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    size: str | None = None,
    color: str | None = None,
    hover_name: str | None = None,
    titulo: str = "",
    x_label: str = "",
    y_label: str = "",
) -> go.Figure:
    """
    Gráfico de dispersión.
    """
    if df.empty:
        return _empty_fig(titulo)

    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size,
        color=color,
        hover_name=hover_name,
        title=titulo,
        color_discrete_sequence=COLOR_SEQUENCE,
        color_continuous_scale=HEATMAP_COLORSCALE,
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis_title=x_label or x,
        yaxis_title=y_label or y,
        title=dict(font=dict(size=14, color=COLORS["primary"]), x=0),
    )
    return fig


def _empty_fig(titulo: str = "") -> go.Figure:
    """Figura vacía con mensaje informativo."""
    fig = go.Figure()
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=titulo, font=dict(size=14, color=COLORS["primary"]), x=0),
        annotations=[
            dict(
                text="Sin datos disponibles para el período seleccionado.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=13, color=COLORS["neutral"]),
            )
        ],
    )
    return fig
