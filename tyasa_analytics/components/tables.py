"""
tables.py — Tablas limpias, scrollables y exportables.
Funciones para mostrar DataFrames con formato ejecutivo y botón de descarga.
"""

import io
import streamlit as st
import pandas as pd
from config import COLORS


def tabla_ejecutiva(
    df: pd.DataFrame,
    titulo: str = "",
    height: int = 400,
    col_formatos: dict | None = None,
    key: str = "tabla",
) -> None:
    """
    Muestra un DataFrame con estilos ejecutivos y opción de exportar.

    Args:
        df: DataFrame a mostrar.
        titulo: título sobre la tabla.
        height: altura en píxeles.
        col_formatos: dict {columna: formato} para formatear columnas numéricas.
                      Ejemplo: {"PESO_TON": "{:,.1f}", "PCT": "{:.1f}%"}
        key: clave única para el widget de descarga.
    """
    if df is None or df.empty:
        st.info("Sin datos para mostrar.")
        return

    if titulo:
        st.markdown(
            f"<h5 style='color:{COLORS['primary']};margin-bottom:6px;'>{titulo}</h5>",
            unsafe_allow_html=True,
        )

    # Aplicar formatos
    df_display = df.copy()
    if col_formatos:
        for col, fmt in col_formatos.items():
            if col in df_display.columns:
                try:
                    df_display[col] = df_display[col].apply(
                        lambda v: fmt.format(v) if pd.notna(v) else ""
                    )
                except Exception:
                    pass

    st.dataframe(
        df_display,
        height=height,
        width="stretch",
        hide_index=True,
    )

    # Botón de descarga
    _boton_descarga(df, key=key)


def _boton_descarga(df: pd.DataFrame, key: str = "export", label: str = "⬇ Exportar Excel") -> None:
    """Renderiza botón para descargar DataFrame como Excel."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos")
    buffer.seek(0)

    st.download_button(
        label=label,
        data=buffer,
        file_name=f"{key}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"dl_{key}",
    )


def tabla_clasificacion_abc(df_abc: pd.DataFrame, key: str = "abc") -> None:
    """
    Tabla formateada específica para clasificación ABC.
    """
    if df_abc.empty:
        st.info("Sin datos para clasificación ABC.")
        return

    col_formatos = {
        "PESO_TON": "{:,.1f}",
        "PCT": "{:.2f}%",
        "PCT_ACUM": "{:.2f}%",
    }

    # Colorear columna CLASE
    def _estilo_clase(val):
        colores = {"A": "#1B3A5C", "B": "#4A7BA7", "C": "#8BA7BF"}
        bg = colores.get(str(val), "white")
        return f"background-color:{bg};color:white;font-weight:bold;text-align:center;"

    df_display = df_abc.copy()
    for col, fmt in col_formatos.items():
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda v: fmt.format(v) if pd.notna(v) else ""
            )

    st.dataframe(
        df_display,
        height=420,
        width="stretch",
        hide_index=True,
    )
    _boton_descarga(df_abc, key=key, label="⬇ Exportar clasificación ABC")


def tabla_metricas(metricas: dict, titulo: str = "Métricas del modelo") -> None:
    """
    Muestra un dict de métricas como tabla compacta de dos columnas.

    Args:
        metricas: dict {nombre: valor}.
        titulo: título de la sección.
    """
    if not metricas:
        return

    st.markdown(
        f"<h5 style='color:{COLORS['primary']};margin-bottom:6px;'>{titulo}</h5>",
        unsafe_allow_html=True,
    )

    cols = st.columns(len(metricas))
    for col, (nombre, valor) in zip(cols, metricas.items()):
        with col:
            st.markdown(
                f"""
                <div style='
                    background:{COLORS["surface"]};
                    border:1px solid #E5E7EB;
                    border-top:3px solid {COLORS["secondary"]};
                    border-radius:6px;
                    padding:12px 14px;
                    text-align:center;
                '>
                    <div style='color:{COLORS["text_light"]};font-size:0.76rem;font-weight:600;text-transform:uppercase;'>{nombre}</div>
                    <div style='color:{COLORS["primary"]};font-size:1.4rem;font-weight:700;'>{valor}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
