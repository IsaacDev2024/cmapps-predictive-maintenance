# core/helpers.py

import numpy as np
import pandas as pd
import streamlit as st


def sanitize_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitiza un DataFrame para que sea compatible con PyArrow/Streamlit.
    Convierte tipos de datos problemáticos a formatos compatibles.
    
    Args:
        df: DataFrame a sanitizar
    
    Returns:
        DataFrame con tipos de datos compatibles con PyArrow
    """
    df_clean = df.copy()
    
    for col in df_clean.columns:
        # Convertir columnas object a string si contienen datos mixtos
        if df_clean[col].dtype == 'object':
            try:
                # Intentar convertir a numérico primero
                df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                # Si sigue siendo object, convertir a string
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].astype(str)
            except:
                df_clean[col] = df_clean[col].astype(str)
        
        # Manejar tipos datetime con timezone
        elif pd.api.types.is_datetime64tz_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].dt.tz_localize(None)
        
        # Convertir tipos complejos a float64
        elif df_clean[col].dtype == 'complex':
            df_clean[col] = df_clean[col].astype('float64')
    
    return df_clean


#
def detect_sensor_columns(df, exclude=None) -> list[str]:
    """
    Detecta columnas numéricas que son sensores, excluyendo columnas especificadas.
    
    Args:
        df: DataFrame
        exclude: Set o lista de nombres de columnas a excluir
    
    Returns:
        Lista de nombres de columnas de sensores
    """
    if exclude is None:
        exclude = {"unit_id", "time_cycles", "RUL", "rul", 
                  "motor", "ciclo",
                  "config1", "config2", "config3",
                  "op_setting_1", "op_setting_2", "op_setting_3"}
    
    return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]


# Validaciones generales
def ensure_columns(df: pd.DataFrame, cols: list[str]):
    """Verifica que el DataFrame contenga las columnas requeridas."""
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        st.error(f"Faltan columnas requeridas: {', '.join(faltan)}")
        st.stop()


def stop_if_empty(df: pd.DataFrame, msg: str = "No hay datos disponibles."):
    """Detiene la ejecución si el DataFrame está vacío."""
    if df.empty:
        st.info(msg)
        st.stop()


# Filtrado y selección

def get_unit_df(df: pd.DataFrame, unit_id: int | str) -> pd.DataFrame:
    """
    Retorna los registros correspondientes a una unidad específica.
    Se asume que la columna de identificación es 'unit_id'.
    """
    if "unit_id" not in df.columns:
        st.error("La columna 'unit_id' no existe en el DataFrame.")
        st.stop()
    return df[df["unit_id"] == unit_id]


def filter_df(df: pd.DataFrame,
              unit_id: int | str | None = None,
              sensor: str | None = None,
              date_range: tuple | None = None,
              time_col: str = "time_cycles") -> pd.DataFrame:
    """
    Aplica filtros genéricos al DataFrame.
    - unit_id: filtra por unidad
    - sensor: filtra por columna (si existe)
    - date_range: tupla (inicio, fin) para limitar por tiempo
    """
    dff = df.copy()

    if unit_id is not None and "unit_id" in dff.columns:
        dff = dff[dff["unit_id"] == unit_id]

    if sensor is not None and sensor in dff.columns:
        # mantiene solo las columnas relevantes
        dff = dff[["unit_id", time_col, sensor]]

    if date_range is not None and len(date_range) == 2:
        ini, fin = date_range
        if time_col in dff.columns:
            dff = dff[(dff[time_col] >= ini) & (dff[time_col] <= fin)]

    return dff


def unique_sorted(df: pd.DataFrame, col: str) -> list:
    """Devuelve los valores únicos ordenados de una columna."""
    return sorted(df[col].dropna().unique().tolist())


# Utilidades de procesamiento y alineación


def pick_unit_key(stationary_obj, unit_id):
    if stationary_obj is None or not isinstance(stationary_obj, dict):
        return None
    if unit_id in stationary_obj:
        return unit_id
    if str(unit_id) in stationary_obj:
        return str(unit_id)
    if int(unit_id) in stationary_obj:
        return int(unit_id)
    return None


def pick_sensor_key(unit_map: dict, sensor_name: str) -> str | None:
    """
    Encuentra la clave correspondiente al nombre del sensor
    dentro del subdiccionario de una unidad.
    """
    if unit_map is None:
        return None

    keys = [str(k) for k in unit_map.keys()]
    matches = [k for k in keys if sensor_name in k]
    return matches[0] if matches else None


def align_x(x_series: pd.Series, stationary_length: int) -> pd.Series:
    """
    Genera una secuencia de tiempo ajustada para los datos estacionarios.
    Si el número de puntos es diferente, recorta o rellena.
    """
    x = np.arange(len(x_series))
    if len(x) == stationary_length:
        return pd.Series(x)
    elif len(x) > stationary_length:
        return pd.Series(x[:stationary_length])
    else:
        return pd.Series(np.arange(stationary_length)[:len(x)])

# Agregaciones y cálculos


def build_lifetable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una tabla con el ciclo máximo de cada unidad.
    Útil para mostrar el ciclo de vida o duración de las unidades.
    """
    ensure_columns(df, ["unit_id", "time_cycles"])
    tabla = (
        df.groupby("unit_id")["time_cycles"]
        .max()
        .reset_index()
        .rename(columns={"time_cycles": "ciclo_maximo"})
    )
    tabla = tabla.sort_values("unit_id").reset_index(drop=True)
    return tabla


def summarize_unit(df_unit: pd.DataFrame, sensor: str) -> dict:
    """
    Calcula métricas descriptivas simples para una unidad y un sensor específico.
    Retorna un diccionario con resultados listos para mostrar.
    """
    stats = {}
    if sensor in df_unit.columns:
        serie = df_unit[sensor].dropna()
        stats["min"] = float(serie.min())
        stats["max"] = float(serie.max())
        stats["mean"] = float(serie.mean())
        stats["std"] = float(serie.std())
        stats["count"] = int(len(serie))
    return stats

# Ui Helpers

def info_missing_stationary(unit_id):
    """Muestra un aviso si no existe data estacionaria para una unidad."""
    st.warning(f"No se encontró información estacionaria para la unidad {unit_id}.")


def describe_filters(params: dict):
    """Muestra un resumen textual de los filtros activos (opcional)."""
    active = [f"{k}: {v}" for k, v in params.items() if v is not None]
    if active:
        st.caption("Filtros activos → " + " | ".join(active))