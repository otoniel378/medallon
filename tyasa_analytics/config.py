"""
config.py — Configuración global de tyasa_analytics.
Centraliza rutas, nombres de tablas, paleta de colores y parámetros.
"""

import os

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta a la base de datos SQLite. Cambia el nombre si es diferente.
DB_PATH = os.path.join(BASE_DIR, "data", "tyasa.db")
# Si usas otro nombre de archivo .db, cámbialo aquí:

EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ---------------------------------------------------------------------------
# Nombre de la app
# ---------------------------------------------------------------------------
APP_NAME = "TYASA Analytics"
APP_SUBTITLE = "Inteligencia Comercial — Aceros Planos Negros"
APP_ICON = "🔩"

# ---------------------------------------------------------------------------
# Tablas SQLite
# ---------------------------------------------------------------------------
TABLES = {
    "bronze_raw": "Base 1926",
    "silver_ventas": "ventas_limpias",
    "gold_demanda_cliente": "gold_demanda_cliente",
    "gold_demanda_producto": "gold_demanda_producto",
    "gold_demanda_mensual": "gold_demanda_mensual",
    "gold_cliente_producto": "gold_cliente_producto",
    "gold_demanda_proceso": "gold_demanda_proceso",
}

# ---------------------------------------------------------------------------
# Columnas esperadas por tabla
# ---------------------------------------------------------------------------
COLS_VENTAS_LIMPIAS = [
    "FECHAEMB",
    "CLIENTE",
    "PRODUCTO_ORIGINAL",
    "PRODUCTO_LIMPIO",
    "CALIBRE",
    "ANCHO",
    "PESO_KG",
    "PESO_TON",
    "PROCESO",
]

# ---------------------------------------------------------------------------
# Paleta de colores corporativa TYASA
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#1B3A5C",        # Azul oscuro corporativo
    "secondary": "#4A7BA7",      # Azul acero medio
    "accent": "#8BA7BF",         # Azul acero claro
    "neutral": "#6B7280",        # Gris neutro
    "background": "#F4F6F9",     # Fondo claro
    "surface": "#FFFFFF",        # Superficie blanca
    "text": "#1F2937",           # Texto oscuro
    "text_light": "#6B7280",     # Texto secundario
    "success": "#2E7D32",        # Verde éxito
    "warning": "#F57C00",        # Naranja alerta
    "danger": "#C62828",         # Rojo peligro
}

# Secuencia de colores para gráficos — paleta diversa industrial
COLOR_SEQUENCE = [
    "#1B3A5C",   # Azul oscuro corporativo (primario)
    "#E05C2D",   # Naranja acero
    "#2D8A5C",   # Verde esmeralda
    "#9B59B6",   # Violeta
    "#D4A017",   # Dorado industrial
    "#00838F",   # Teal
    "#C0392B",   # Rojo
    "#2980B9",   # Azul celeste
    "#27AE60",   # Verde brillante
    "#D35400",   # Naranja oscuro
    "#8E44AD",   # Violeta oscuro
    "#16A085",   # Verde azulado
    "#F39C12",   # Ámbar
    "#1ABC9C",   # Menta
    "#6C3483",   # Púrpura profundo
]

# Escala de calor para heatmaps (blanco → azul profundo)
HEATMAP_COLORSCALE = [
    [0.0,  "#F8FAFC"],
    [0.15, "#D6E8F5"],
    [0.40, "#7BAFD4"],
    [0.70, "#2E6FA3"],
    [1.0,  "#0D2137"],
]

# ---------------------------------------------------------------------------
# Parámetros analíticos
# ---------------------------------------------------------------------------
FORECAST_HORIZON_DEFAULT = 6       # meses a pronosticar por defecto
FORECAST_HORIZON_MAX = 24          # horizonte máximo permitido
MIN_PERIODS_FORECAST = 12          # mínimo de períodos para hacer forecast
PARETO_THRESHOLD_A = 0.80          # acumulado para clase A
PARETO_THRESHOLD_B = 0.95          # acumulado para clase B

# ---------------------------------------------------------------------------
# Familias de producto (escala futura)
# ---------------------------------------------------------------------------
FAMILIAS = {
    "aceros_planos_negros": "Aceros Planos Negros",
    # "galvanizados": "Galvanizados",       # habilitación futura
    # "formados": "Formados",               # habilitación futura
}

FAMILIA_ACTIVA = "aceros_planos_negros"
