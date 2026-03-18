"""
mix_productos.py — Análisis de mix de productos y oportunidades de cross-sell.
"""

import pandas as pd
import numpy as np
from itertools import combinations


def participacion_por_familia(df_producto: pd.DataFrame, col_prod: str = "PRODUCTO_LIMPIO", col_val: str = "PESO_TON") -> pd.DataFrame:
    """
    Calcula participación porcentual de cada familia/producto.

    Returns:
        DataFrame con PRODUCTO_LIMPIO, PESO_TON, PCT.
    """
    if df_producto.empty:
        return pd.DataFrame()

    agg = (
        df_producto.groupby(col_prod, as_index=False)[col_val]
        .sum()
        .sort_values(col_val, ascending=False)
        .reset_index(drop=True)
    )
    total = agg[col_val].sum()
    agg["PCT"] = (agg[col_val] / total * 100).round(1) if total > 0 else 0.0
    agg["PCT_ACUM"] = agg["PCT"].cumsum().round(1)
    return agg


def n_familias_por_cliente(df_cliente_producto: pd.DataFrame) -> pd.DataFrame:
    """
    Número de familias/productos distintos por cliente.

    Returns:
        DataFrame con CLIENTE, N_PRODUCTOS, PESO_TON.
    """
    if df_cliente_producto.empty:
        return pd.DataFrame()

    return (
        df_cliente_producto.groupby("CLIENTE", as_index=False)
        .agg(N_PRODUCTOS=("PRODUCTO_LIMPIO", "nunique"), PESO_TON=("PESO_TON", "sum"))
        .sort_values("N_PRODUCTOS", ascending=False)
        .reset_index(drop=True)
    )


def tabla_coocurrencia(df_cliente_producto: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una tabla de co-ocurrencia de productos por cliente.
    Cada celda [prod_A, prod_B] = número de clientes que compraron ambos productos.

    Returns:
        pd.DataFrame cuadrado: índice = columna = PRODUCTO_LIMPIO.
    """
    if df_cliente_producto.empty:
        return pd.DataFrame()

    # Clientes que compraron cada producto
    clientes_por_prod = (
        df_cliente_producto.groupby("PRODUCTO_LIMPIO")["CLIENTE"]
        .apply(set)
        .to_dict()
    )

    productos = sorted(clientes_por_prod.keys())
    matrix = pd.DataFrame(0, index=productos, columns=productos)

    for pa, pb in combinations(productos, 2):
        n = len(clientes_por_prod[pa] & clientes_por_prod[pb])
        matrix.loc[pa, pb] = n
        matrix.loc[pb, pa] = n

    # Diagonal: clientes únicos del propio producto
    for p in productos:
        matrix.loc[p, p] = len(clientes_por_prod[p])

    return matrix


def combinaciones_frecuentes(df_cliente_producto: pd.DataFrame, min_clientes: int = 2) -> pd.DataFrame:
    """
    Devuelve pares de productos frecuentemente comprados juntos.

    Args:
        df_cliente_producto: DataFrame con CLIENTE y PRODUCTO_LIMPIO.
        min_clientes: mínimo de clientes para incluir el par.

    Returns:
        DataFrame con PRODUCTO_A, PRODUCTO_B, N_CLIENTES.
    """
    if df_cliente_producto.empty:
        return pd.DataFrame()

    clientes_por_prod = (
        df_cliente_producto.groupby("PRODUCTO_LIMPIO")["CLIENTE"]
        .apply(set)
        .to_dict()
    )

    rows = []
    productos = sorted(clientes_por_prod.keys())
    for pa, pb in combinations(productos, 2):
        n = len(clientes_por_prod[pa] & clientes_por_prod[pb])
        if n >= min_clientes:
            rows.append({"PRODUCTO_A": pa, "PRODUCTO_B": pb, "N_CLIENTES": n})

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("N_CLIENTES", ascending=False)
        .reset_index(drop=True)
    )


def oportunidades_crosssell(
    df_cliente_producto: pd.DataFrame,
    min_soporte: float = 0.1,
) -> pd.DataFrame:
    """
    Identifica clientes que aún no compran un producto que sus similares sí compran.
    Usa lógica de soporte: producto comprado por ≥ min_soporte% de clientes activos.

    Args:
        df_cliente_producto: datos de clientes y productos.
        min_soporte: fracción mínima de clientes que deben comprar el producto.

    Returns:
        DataFrame con CLIENTE, PRODUCTO_SUGERIDO, N_OTROS_CLIENTES_LO_COMPRAN.
    """
    if df_cliente_producto.empty:
        return pd.DataFrame()

    total_clientes = df_cliente_producto["CLIENTE"].nunique()
    umbral = int(total_clientes * min_soporte)
    umbral = max(umbral, 2)

    # Productos con soporte suficiente
    popularidad = (
        df_cliente_producto.groupby("PRODUCTO_LIMPIO")["CLIENTE"]
        .nunique()
        .reset_index()
        .rename(columns={"CLIENTE": "N_CLIENTES"})
    )
    populares = popularidad[popularidad["N_CLIENTES"] >= umbral]["PRODUCTO_LIMPIO"].tolist()

    if not populares:
        return pd.DataFrame()

    # Productos ya comprados por cada cliente
    ya_comprados = (
        df_cliente_producto.groupby("CLIENTE")["PRODUCTO_LIMPIO"]
        .apply(set)
        .to_dict()
    )

    rows = []
    for cliente, comprados in ya_comprados.items():
        faltantes = set(populares) - comprados
        for prod in faltantes:
            n = popularidad[popularidad["PRODUCTO_LIMPIO"] == prod]["N_CLIENTES"].values[0]
            rows.append({
                "CLIENTE": cliente,
                "PRODUCTO_SUGERIDO": prod,
                "N_CLIENTES_LO_COMPRAN": int(n),
            })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["CLIENTE", "N_CLIENTES_LO_COMPRAN"], ascending=[True, False])
        .reset_index(drop=True)
    )
