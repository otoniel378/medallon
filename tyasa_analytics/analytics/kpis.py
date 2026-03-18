"""
kpis.py — Cálculo de KPIs ejecutivos globales.
Consume tablas Gold y devuelve métricas agregadas.
"""

import pandas as pd
from dataclasses import dataclass


@dataclass
class KPIResumen:
    """Estructura de KPIs del Resumen Ejecutivo."""
    toneladas_totales: float
    clientes_activos: int
    productos_activos: int
    ticket_promedio: float       # toneladas promedio por cliente
    variacion_mom: float | None  # variación mes actual vs anterior (%)
    top_cliente: str
    top_producto: str


def calcular_kpis_resumen(
    df_cliente: pd.DataFrame,
    df_producto: pd.DataFrame,
    df_mensual: pd.DataFrame,
) -> KPIResumen:
    """
    Calcula los KPIs del resumen ejecutivo a partir de las tablas Gold.

    Args:
        df_cliente: gold_demanda_cliente
        df_producto: gold_demanda_producto
        df_mensual: gold_demanda_mensual (con columna PERIODO)

    Returns:
        KPIResumen con los indicadores calculados.
    """
    # Toneladas totales
    toneladas = float(df_cliente["PESO_TON"].sum()) if not df_cliente.empty else 0.0

    # Clientes activos
    clientes_activos = (
        df_cliente["CLIENTE"].nunique() if not df_cliente.empty else 0
    )

    # Productos activos
    productos_activos = (
        df_producto["PRODUCTO_LIMPIO"].nunique() if not df_producto.empty else 0
    )

    # Ticket promedio (toneladas por cliente)
    ticket_promedio = (
        toneladas / clientes_activos if clientes_activos > 0 else 0.0
    )

    # Variación MoM del último mes vs penúltimo
    variacion_mom = None
    if not df_mensual.empty and "PERIODO" in df_mensual.columns and "PESO_TON" in df_mensual.columns:
        serie = df_mensual.sort_values("PERIODO").tail(2)
        if len(serie) == 2:
            anterior, actual = serie["PESO_TON"].iloc[0], serie["PESO_TON"].iloc[1]
            if anterior > 0:
                variacion_mom = round((actual - anterior) / anterior * 100, 1)

    # Top cliente
    top_cliente = ""
    if not df_cliente.empty and "CLIENTE" in df_cliente.columns:
        idx = df_cliente["PESO_TON"].idxmax()
        top_cliente = df_cliente.loc[idx, "CLIENTE"]

    # Top producto
    top_producto = ""
    if not df_producto.empty and "PRODUCTO_LIMPIO" in df_producto.columns:
        idx = df_producto["PESO_TON"].idxmax()
        top_producto = df_producto.loc[idx, "PRODUCTO_LIMPIO"]

    return KPIResumen(
        toneladas_totales=round(toneladas, 1),
        clientes_activos=clientes_activos,
        productos_activos=productos_activos,
        ticket_promedio=round(ticket_promedio, 1),
        variacion_mom=variacion_mom,
        top_cliente=top_cliente,
        top_producto=top_producto,
    )


def calcular_top_n(df: pd.DataFrame, col_dim: str, col_val: str = "PESO_TON", n: int = 10) -> pd.DataFrame:
    """
    Calcula el top N de una dimensión ordenado por valor descendente.

    Args:
        df: DataFrame fuente.
        col_dim: columna de dimensión (ej. CLIENTE, PRODUCTO_LIMPIO).
        col_val: columna de valor (ej. PESO_TON).
        n: número de registros.

    Returns:
        DataFrame con las n filas superiores.
    """
    if df.empty or col_dim not in df.columns or col_val not in df.columns:
        return pd.DataFrame()
    return (
        df[[col_dim, col_val]]
        .groupby(col_dim, as_index=False)[col_val]
        .sum()
        .sort_values(col_val, ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def calcular_participacion(df: pd.DataFrame, col_dim: str, col_val: str = "PESO_TON") -> pd.DataFrame:
    """
    Agrega participación porcentual a un DataFrame agrupado.

    Args:
        df: DataFrame con col_dim y col_val.
        col_dim: dimensión de agrupación.
        col_val: columna de valor.

    Returns:
        DataFrame con columna adicional PCT.
    """
    if df.empty:
        return df
    agg = (
        df[[col_dim, col_val]]
        .groupby(col_dim, as_index=False)[col_val]
        .sum()
    )
    total = agg[col_val].sum()
    agg["PCT"] = (agg[col_val] / total * 100).round(1) if total > 0 else 0.0
    return agg.sort_values(col_val, ascending=False).reset_index(drop=True)
