"""
series_tiempo.py — Análisis de series de tiempo de demanda.
Calcula variaciones, volatilidad, heatmaps y rankings de estabilidad.
"""

import pandas as pd
import numpy as np


def preparar_serie_mensual(df_mensual: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara y ordena la serie mensual. Agrega columnas ANIO y MES si faltan.

    Args:
        df_mensual: gold_demanda_mensual con columna PERIODO y PESO_TON.

    Returns:
        DataFrame ordenado con PERIODO, PESO_TON, ANIO, MES.
    """
    if df_mensual.empty or "PERIODO" not in df_mensual.columns:
        return pd.DataFrame()

    df = df_mensual.copy().sort_values("PERIODO").reset_index(drop=True)

    if "ANIO" not in df.columns:
        df["ANIO"] = df["PERIODO"].dt.year
    if "MES" not in df.columns:
        df["MES"] = df["PERIODO"].dt.month

    df["ANIO"] = df["ANIO"].astype(int)
    df["MES"] = df["MES"].astype(int)
    return df


def calcular_variacion_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variación mensual (MoM) porcentual a la serie.

    Args:
        df: serie mensual ordenada con PESO_TON.

    Returns:
        DataFrame con columnas adicionales: VAR_MOM, VAR_MOM_PCT.
    """
    if df.empty:
        return df

    df = df.copy()
    df["VAR_MOM"] = df["PESO_TON"].diff()
    df["VAR_MOM_PCT"] = df["PESO_TON"].pct_change() * 100
    df["VAR_MOM_PCT"] = df["VAR_MOM_PCT"].round(1)
    return df


def calcular_volatilidad(df: pd.DataFrame, col_dim: str | None = None, col_val: str = "PESO_TON") -> pd.DataFrame:
    """
    Calcula coeficiente de variación (CV) por dimensión (familia, cliente) o global.

    Args:
        df: DataFrame con col_dim (opcional) y col_val.
        col_dim: columna de agrupación (ej. PRODUCTO_LIMPIO). Si es None, calcula global.
        col_val: columna de valor.

    Returns:
        DataFrame con DIMENSION, MEDIA, STD, CV (%), N.
    """
    if df.empty:
        return pd.DataFrame()

    if col_dim and col_dim in df.columns:
        grupos = df.groupby(col_dim)[col_val]
        result = grupos.agg(
            MEDIA="mean",
            STD="std",
            N="count",
        ).reset_index()
        result.columns = ["DIMENSION", "MEDIA", "STD", "N"]
    else:
        result = pd.DataFrame([{
            "DIMENSION": "Global",
            "MEDIA": df[col_val].mean(),
            "STD": df[col_val].std(),
            "N": len(df),
        }])

    result["CV"] = (result["STD"] / result["MEDIA"] * 100).round(1).fillna(0)
    result = result.sort_values("CV").reset_index(drop=True)
    result["ESTABILIDAD"] = result["CV"].apply(_clasificar_estabilidad)
    return result


def _clasificar_estabilidad(cv: float) -> str:
    """Clasifica estabilidad según CV."""
    if cv < 20:
        return "Alta"
    elif cv < 40:
        return "Media"
    return "Baja"


def construir_heatmap_mes_anio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la tabla pivotada MES × AÑO con toneladas para el heatmap.

    Args:
        df: serie mensual con columnas ANIO, MES, PESO_TON.

    Returns:
        DataFrame pivotado: index=MES, columns=ANIO, values=PESO_TON.
    """
    if df.empty or "ANIO" not in df.columns or "MES" not in df.columns:
        return pd.DataFrame()

    pivot = df.pivot_table(
        index="MES",
        columns="ANIO",
        values="PESO_TON",
        aggfunc="sum",
        fill_value=0,
    )
    pivot.index = [_nombre_mes(m) for m in pivot.index]
    return pivot


def _nombre_mes(mes: int) -> str:
    nombres = {
        1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Ago",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
    }
    return nombres.get(mes, str(mes))


def ranking_estabilidad(df_cliente_producto: pd.DataFrame | None = None,
                         df_mensual: pd.DataFrame | None = None,
                         col_dim: str = "PRODUCTO_LIMPIO",
                         col_val: str = "PESO_TON") -> pd.DataFrame:
    """
    Genera ranking de estabilidad de demanda por dimensión.

    Args:
        df_cliente_producto: datos a nivel transaccional o agrupados por dimensión.
        df_mensual: serie mensual (si se usa para ranking global).
        col_dim: dimensión de agrupación.
        col_val: métrica de volumen.

    Returns:
        DataFrame con DIMENSION, CV, ESTABILIDAD, MEDIA.
    """
    df_input = df_cliente_producto if df_cliente_producto is not None else df_mensual
    if df_input is None or df_input.empty:
        return pd.DataFrame()

    return calcular_volatilidad(df_input, col_dim=col_dim if col_dim in df_input.columns else None, col_val=col_val)


def top_afectados_variacion(
    df: pd.DataFrame,
    col_dim: str,
    col_periodo: str = "PERIODO",
    col_val: str = "PESO_TON",
    n: int = 10,
) -> pd.DataFrame:
    """
    Devuelve las N dimensiones con mayor variación absoluta entre los
    dos últimos periodos disponibles.

    Args:
        df: DataFrame con col_dim, col_periodo, col_val.
        col_dim: columna de dimensión (producto, proceso, cliente).
        n: número de elementos a devolver.

    Returns:
        DataFrame con DIMENSION, ACTUAL, ANTERIOR, VAR_ABS, VAR_PCT.
    """
    if df.empty or col_dim not in df.columns or col_periodo not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df[col_periodo] = pd.to_datetime(df[col_periodo], errors="coerce")
    periodos = sorted(df[col_periodo].dropna().unique())
    if len(periodos) < 2:
        return pd.DataFrame()

    p_actual = periodos[-1]
    p_anterior = periodos[-2]

    mes_a = df[df[col_periodo] == p_actual].groupby(col_dim)[col_val].sum()
    mes_p = df[df[col_periodo] == p_anterior].groupby(col_dim)[col_val].sum()

    merged = pd.DataFrame({"ACTUAL": mes_a, "ANTERIOR": mes_p}).fillna(0).reset_index()
    merged.columns = ["DIMENSION", "ACTUAL", "ANTERIOR"]
    merged["VAR_ABS"] = (merged["ACTUAL"] - merged["ANTERIOR"]).round(1)
    denom = merged["ANTERIOR"].replace(0, float("nan"))
    merged["VAR_PCT"] = ((merged["VAR_ABS"] / denom) * 100).round(1)

    merged = merged.reindex(merged["VAR_ABS"].abs().sort_values(ascending=False).index)
    return merged.head(n).reset_index(drop=True)


def serie_por_dimension(df: pd.DataFrame, col_dim: str, col_periodo: str = "PERIODO", col_val: str = "PESO_TON") -> pd.DataFrame:
    """
    Construye serie mensual agrupada por una dimensión (ej. PRODUCTO_LIMPIO).

    Args:
        df: DataFrame con col_dim, col_periodo, col_val.
        col_dim: dimensión de agrupación.
        col_periodo: columna de periodo (datetime).
        col_val: métrica.

    Returns:
        DataFrame con PERIODO, col_dim, col_val (suma).
    """
    if df.empty:
        return pd.DataFrame()

    if col_periodo not in df.columns:
        return pd.DataFrame()

    result = (
        df.groupby([col_periodo, col_dim], as_index=False)[col_val]
        .sum()
        .sort_values([col_periodo, col_dim])
        .reset_index(drop=True)
    )
    return result
