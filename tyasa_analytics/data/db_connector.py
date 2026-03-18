"""
db_connector.py — Gestión de conexión a SQLite.
Proporciona una conexión reutilizable con manejo de errores.
"""

import sqlite3
import os
import pandas as pd
import streamlit as st
from config import DB_PATH


def get_connection() -> sqlite3.Connection:
    """
    Abre y devuelve una conexión SQLite.

    Returns:
        sqlite3.Connection: conexión activa a la base de datos.

    Raises:
        FileNotFoundError: si el archivo .db no existe.
        sqlite3.Error: si hay un problema al conectar.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"No se encontró la base de datos en: {DB_PATH}\n"
            "Verifica que el archivo SQLite esté en la carpeta 'data/' "
            "y que su nombre coincida con DB_PATH en config.py."
        )
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error al conectar con SQLite: {e}") from e


def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """
    Ejecuta una consulta SQL y devuelve un DataFrame.

    Args:
        sql: sentencia SQL a ejecutar.
        params: parámetros opcionales para la consulta parametrizada.

    Returns:
        pd.DataFrame con los resultados.
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn, params=params)
        return df
    except Exception as e:
        raise RuntimeError(f"Error ejecutando query:\n{sql}\n\nDetalle: {e}") from e
    finally:
        conn.close()


def list_tables() -> list[str]:
    """Devuelve la lista de tablas disponibles en la base de datos."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


@st.cache_resource(show_spinner=False)
def get_cached_connection():
    """
    Conexión cacheada por Streamlit (singleton por sesión).
    Útil para evitar reconexiones repetidas.
    """
    return get_connection()
