"""
loaders.py — Funciones de carga para cada tabla Gold y Silver.
Usa caché de Streamlit para evitar re-consultas innecesarias.
"""

import pandas as pd
import streamlit as st
from data.db_connector import run_query
from config import TABLES


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _safe_load(table_name: str, display_name: str) -> pd.DataFrame:
    """Carga genérica con manejo de errores."""
    try:
        return run_query(f'SELECT * FROM "{table_name}"')
    except Exception as e:
        st.error(f"No se pudo cargar '{display_name}': {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Gold tables
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner="Cargando demanda por cliente…")
def load_gold_demanda_cliente() -> pd.DataFrame:
    """
    Carga gold_demanda_cliente.
    Columnas esperadas: CLIENTE, PESO_TON (y otras métricas agregadas).
    """
    df = _safe_load(TABLES["gold_demanda_cliente"], "gold_demanda_cliente")
    if not df.empty and "PESO_TON" in df.columns:
        df["PESO_TON"] = pd.to_numeric(df["PESO_TON"], errors="coerce").fillna(0)
    return df


@st.cache_data(ttl=600, show_spinner="Cargando demanda por producto…")
def load_gold_demanda_producto() -> pd.DataFrame:
    """
    Carga gold_demanda_producto.
    Columnas esperadas: PRODUCTO_LIMPIO, PESO_TON.
    """
    df = _safe_load(TABLES["gold_demanda_producto"], "gold_demanda_producto")
    if not df.empty and "PESO_TON" in df.columns:
        df["PESO_TON"] = pd.to_numeric(df["PESO_TON"], errors="coerce").fillna(0)
    return df


@st.cache_data(ttl=600, show_spinner="Cargando serie mensual…")
def load_gold_demanda_mensual_total() -> pd.DataFrame:
    """
    Serie mensual TOTAL (agrega todos los productos por período).
    Columnas: PERIODO (datetime), PESO_TON, ANIO, MES.
    """
    df = _safe_load(TABLES["gold_demanda_mensual"], "gold_demanda_mensual")
    if df.empty:
        return df

    df["PESO_TON"] = pd.to_numeric(df["PESO_TON"], errors="coerce").fillna(0)
    df["PERIODO"] = pd.to_datetime(df["PERIODO"], errors="coerce")

    agg = (
        df.groupby("PERIODO", as_index=False)["PESO_TON"]
        .sum()
        .sort_values("PERIODO")
        .reset_index(drop=True)
    )
    agg["ANIO"] = agg["PERIODO"].dt.year
    agg["MES"] = agg["PERIODO"].dt.month
    return agg


@st.cache_data(ttl=600, show_spinner="Cargando serie mensual…")
def load_gold_demanda_mensual() -> pd.DataFrame:
    """
    Carga gold_demanda_mensual.
    Columnas esperadas: ANIO, MES, PESO_TON (y opcionalmente PERIODO como date).
    Crea columna PERIODO (datetime) si no existe.
    """
    df = _safe_load(TABLES["gold_demanda_mensual"], "gold_demanda_mensual")
    if df.empty:
        return df

    if "PESO_TON" in df.columns:
        df["PESO_TON"] = pd.to_numeric(df["PESO_TON"], errors="coerce").fillna(0)

    # Construir columna PERIODO si existen ANIO y MES
    if "PERIODO" not in df.columns and "ANIO" in df.columns and "MES" in df.columns:
        df["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce")
        df["MES"] = pd.to_numeric(df["MES"], errors="coerce")
        df["PERIODO"] = pd.to_datetime(
            df["ANIO"].astype(int).astype(str) + "-"
            + df["MES"].astype(int).astype(str).str.zfill(2) + "-01"
        )
    elif "PERIODO" in df.columns:
        df["PERIODO"] = pd.to_datetime(df["PERIODO"], errors="coerce")

    return df.sort_values("PERIODO") if "PERIODO" in df.columns else df


@st.cache_data(ttl=600, show_spinner="Cargando matriz cliente-producto…")
def load_gold_cliente_producto() -> pd.DataFrame:
    """
    Carga gold_cliente_producto.
    Columnas esperadas: CLIENTE, PRODUCTO_LIMPIO, PESO_TON.
    """
    df = _safe_load(TABLES["gold_cliente_producto"], "gold_cliente_producto")
    if not df.empty and "PESO_TON" in df.columns:
        df["PESO_TON"] = pd.to_numeric(df["PESO_TON"], errors="coerce").fillna(0)
    return df


@st.cache_data(ttl=600, show_spinner="Cargando demanda por proceso…")
def load_gold_demanda_proceso() -> pd.DataFrame:
    """
    Carga gold_demanda_proceso.
    Columnas esperadas: PROCESO, PESO_TON.
    """
    df = _safe_load(TABLES["gold_demanda_proceso"], "gold_demanda_proceso")
    if not df.empty and "PESO_TON" in df.columns:
        df["PESO_TON"] = pd.to_numeric(df["PESO_TON"], errors="coerce").fillna(0)
    return df


# ---------------------------------------------------------------------------
# Silver table
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner="Cargando ventas limpias…")
def load_ventas_limpias() -> pd.DataFrame:
    """
    Carga la tabla Silver ventas_limpias con tipos correctos.
    """
    df = _safe_load(TABLES["silver_ventas"], "ventas_limpias")
    if df.empty:
        return df

    if "FECHAEMB" in df.columns:
        df["FECHAEMB"] = pd.to_datetime(df["FECHAEMB"], errors="coerce")
    for col in ["PESO_KG", "PESO_TON"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ["CALIBRE", "ANCHO"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Utilidad: catálogos de filtros
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def get_catalogo_clientes() -> list[str]:
    """Lista ordenada de clientes únicos."""
    df = load_gold_demanda_cliente()
    if df.empty or "CLIENTE" not in df.columns:
        return []
    return sorted(df["CLIENTE"].dropna().unique().tolist())


@st.cache_data(ttl=600, show_spinner=False)
def get_catalogo_productos() -> list[str]:
    """Lista ordenada de productos únicos."""
    df = load_gold_demanda_producto()
    if df.empty or "PRODUCTO_LIMPIO" not in df.columns:
        return []
    return sorted(df["PRODUCTO_LIMPIO"].dropna().unique().tolist())


@st.cache_data(ttl=600, show_spinner=False)
def get_catalogo_procesos() -> list[str]:
    """Lista ordenada de procesos únicos."""
    df = load_gold_demanda_proceso()
    if df.empty or "PROCESO" not in df.columns:
        return []
    return sorted(df["PROCESO"].dropna().unique().tolist())


@st.cache_data(ttl=600, show_spinner=False)
def get_rango_fechas() -> tuple:
    """Devuelve (fecha_min, fecha_max) de la serie mensual."""
    df = load_gold_demanda_mensual()
    if df.empty or "PERIODO" not in df.columns:
        return (None, None)
    return df["PERIODO"].min(), df["PERIODO"].max()


@st.cache_data(ttl=600, show_spinner="Cargando serie mensual por proceso…")
def load_serie_mensual_proceso() -> pd.DataFrame:
    """
    Serie mensual por PROCESO construida desde ventas_limpias.
    Columnas: PERIODO (datetime), PROCESO, PESO_TON.
    """
    df = _safe_load(TABLES["silver_ventas"], "ventas_limpias")
    if df.empty or "PROCESO" not in df.columns or "FECHAEMB" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["FECHAEMB"] = pd.to_datetime(df["FECHAEMB"], errors="coerce")
    df["PESO_TON"] = pd.to_numeric(df.get("PESO_TON", 0), errors="coerce").fillna(0)
    df = df.dropna(subset=["FECHAEMB"])
    df["PERIODO"] = df["FECHAEMB"].dt.to_period("M").dt.to_timestamp()
    result = (
        df.groupby(["PERIODO", "PROCESO"], as_index=False)["PESO_TON"]
        .sum()
        .sort_values(["PERIODO", "PROCESO"])
        .reset_index(drop=True)
    )
    return result


@st.cache_data(ttl=600, show_spinner="Cargando serie mensual por cliente…")
def load_serie_mensual_cliente() -> pd.DataFrame:
    """
    Serie mensual por CLIENTE construida desde ventas_limpias.
    Columnas: PERIODO (datetime), CLIENTE, PESO_TON.
    """
    df = _safe_load(TABLES["silver_ventas"], "ventas_limpias")
    if df.empty or "CLIENTE" not in df.columns or "FECHAEMB" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["FECHAEMB"] = pd.to_datetime(df["FECHAEMB"], errors="coerce")
    df["PESO_TON"] = pd.to_numeric(df.get("PESO_TON", 0), errors="coerce").fillna(0)
    df = df.dropna(subset=["FECHAEMB"])
    df["PERIODO"] = df["FECHAEMB"].dt.to_period("M").dt.to_timestamp()
    result = (
        df.groupby(["PERIODO", "CLIENTE"], as_index=False)["PESO_TON"]
        .sum()
        .sort_values(["PERIODO", "CLIENTE"])
        .reset_index(drop=True)
    )
    return result


@st.cache_data(ttl=600, show_spinner=False)
def get_catalogo_anios() -> list[int]:
    """Lista de años disponibles en ventas_limpias (descendente)."""
    df = _safe_load(TABLES["silver_ventas"], "ventas_limpias")
    if df.empty or "FECHAEMB" not in df.columns:
        return []
    fechas = pd.to_datetime(df["FECHAEMB"], errors="coerce").dropna()
    return sorted(fechas.dt.year.unique().tolist(), reverse=True)
