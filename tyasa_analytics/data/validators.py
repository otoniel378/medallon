"""
validators.py — Validaciones de integridad para DataFrames cargados.
Verifica columnas, vacíos críticos y consistencia mínima.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationResult:
    """Resultado de una validación."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"❌ {e}")
        for w in self.warnings:
            lines.append(f"⚠️ {w}")
        return "\n".join(lines) if lines else "✅ Sin problemas detectados."


def _check_columns(df: pd.DataFrame, required: list[str], table: str, result: ValidationResult):
    """Verifica que las columnas requeridas existan."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        result.add_error(f"[{table}] Columnas faltantes: {missing}")


def _check_empty(df: pd.DataFrame, col: str, table: str, result: ValidationResult, threshold: float = 0.5):
    """Alerta si una columna tiene más de `threshold` de nulos."""
    if col not in df.columns:
        return
    ratio = df[col].isna().mean()
    if ratio > threshold:
        result.add_error(f"[{table}].{col}: {ratio:.0%} de valores nulos (umbral: {threshold:.0%})")
    elif ratio > 0.05:
        result.add_warning(f"[{table}].{col}: {ratio:.1%} de valores nulos")


def validate_ventas_limpias(df: pd.DataFrame) -> ValidationResult:
    """Valida la tabla Silver ventas_limpias."""
    result = ValidationResult(is_valid=True)

    if df is None or df.empty:
        result.add_error("ventas_limpias está vacía o no se cargó.")
        return result

    required = ["FECHAEMB", "CLIENTE", "PRODUCTO_LIMPIO", "PESO_TON", "PROCESO"]
    _check_columns(df, required, "ventas_limpias", result)

    for col in ["FECHAEMB", "CLIENTE", "PESO_TON"]:
        _check_empty(df, col, "ventas_limpias", result)

    # Verificar que PESO_TON sea numérico y positivo
    if "PESO_TON" in df.columns:
        negatives = (df["PESO_TON"] < 0).sum()
        if negatives > 0:
            result.add_warning(f"ventas_limpias.PESO_TON: {negatives} registros con valor negativo.")

    return result


def validate_gold_demanda_cliente(df: pd.DataFrame) -> ValidationResult:
    """Valida gold_demanda_cliente."""
    result = ValidationResult(is_valid=True)

    if df is None or df.empty:
        result.add_error("gold_demanda_cliente está vacía.")
        return result

    _check_columns(df, ["CLIENTE", "PESO_TON"], "gold_demanda_cliente", result)
    _check_empty(df, "CLIENTE", "gold_demanda_cliente", result)
    _check_empty(df, "PESO_TON", "gold_demanda_cliente", result)
    return result


def validate_gold_demanda_producto(df: pd.DataFrame) -> ValidationResult:
    """Valida gold_demanda_producto."""
    result = ValidationResult(is_valid=True)

    if df is None or df.empty:
        result.add_error("gold_demanda_producto está vacía.")
        return result

    _check_columns(df, ["PRODUCTO_LIMPIO", "PESO_TON"], "gold_demanda_producto", result)
    _check_empty(df, "PRODUCTO_LIMPIO", "gold_demanda_producto", result)
    return result


def validate_gold_demanda_mensual(df: pd.DataFrame) -> ValidationResult:
    """Valida gold_demanda_mensual."""
    result = ValidationResult(is_valid=True)

    if df is None or df.empty:
        result.add_error("gold_demanda_mensual está vacía.")
        return result

    _check_columns(df, ["PESO_TON"], "gold_demanda_mensual", result)

    if "PERIODO" not in df.columns and not (
        "ANIO" in df.columns and "MES" in df.columns
    ):
        result.add_error(
            "gold_demanda_mensual: se requiere columna PERIODO o las columnas ANIO + MES."
        )

    _check_empty(df, "PESO_TON", "gold_demanda_mensual", result)

    if "PESO_TON" in df.columns:
        n_rows = len(df)
        if n_rows < 12:
            result.add_warning(
                f"gold_demanda_mensual: solo {n_rows} registros. "
                "El forecasting puede ser poco confiable."
            )
    return result


def validate_gold_cliente_producto(df: pd.DataFrame) -> ValidationResult:
    """Valida gold_cliente_producto."""
    result = ValidationResult(is_valid=True)

    if df is None or df.empty:
        result.add_error("gold_cliente_producto está vacía.")
        return result

    _check_columns(df, ["CLIENTE", "PRODUCTO_LIMPIO", "PESO_TON"], "gold_cliente_producto", result)
    return result


def validate_all(dataframes: dict) -> dict[str, ValidationResult]:
    """
    Ejecuta todas las validaciones.

    Args:
        dataframes: dict con claves 'ventas', 'cliente', 'producto', 'mensual', 'cliente_producto'.

    Returns:
        dict con resultados por tabla.
    """
    validators = {
        "ventas_limpias": validate_ventas_limpias,
        "gold_demanda_cliente": validate_gold_demanda_cliente,
        "gold_demanda_producto": validate_gold_demanda_producto,
        "gold_demanda_mensual": validate_gold_demanda_mensual,
        "gold_cliente_producto": validate_gold_cliente_producto,
    }

    results = {}
    for key, fn in validators.items():
        df = dataframes.get(key, pd.DataFrame())
        results[key] = fn(df)

    return results
