# core/ui.py

import pandas as pd
import streamlit as st
from core import helpers


# sidebar - Controles

# evolution
def controls_evolution(df: pd.DataFrame,
                       stationary_map: dict | None = None,
                       location: str = "sidebar") -> dict:
    """
    Controles de la vista de evolución (ejemplo general).
    Devuelve un diccionario con los parámetros seleccionados.
    """
    ui = st.sidebar if location == "sidebar" else st


    # --- Selector de unidad ---
    if "unit_id" not in df.columns:
        st.error("No se encontró la columna 'unit_id' en los datos.")
        st.stop()
    unidades = sorted(df["unit_id"].dropna().unique())
    unidad = ui.selectbox("Unidad", unidades)

    # --- Selector de sensor ---
    # Detecta columnas tipo sensor automáticamente (excluye configuraciones operacionales)
    sensores = helpers.detect_sensor_columns(df, exclude={'unit_id', 'time_cycles', 'RUL', 'rul', 'motor', 'ciclo', 
                                                           'config1', 'config2', 'config3', 
                                                           'op_setting_1', 'op_setting_2', 'op_setting_3'})
    
    # Crear diccionario con nombres descriptivos para el selectbox
    from core import config
    sensor_options = {}
    for sensor in sensores:
        if sensor in config.SENSOR_SHORT_NAMES:
            # Usar solo el nombre corto del sensor
            label = config.SENSOR_SHORT_NAMES[sensor]
            sensor_options[label] = sensor
        else:
            sensor_options[sensor] = sensor
    
    # Inicializar el estado del sensor si no existe
    if 'selected_sensor_label' not in st.session_state:
        st.session_state.selected_sensor_label = list(sensor_options.keys())[0]
    
    # Obtener el índice del sensor guardado en session_state
    sensor_labels = list(sensor_options.keys())
    default_index = sensor_labels.index(st.session_state.selected_sensor_label) if st.session_state.selected_sensor_label in sensor_labels else 0
    
    # Mostrar selectbox con nombres descriptivos
    selected_label = ui.selectbox("Sensor", sensor_labels, index=default_index, key="evolution_sensor_selector")
    
    # Actualizar el estado global
    st.session_state.selected_sensor_label = selected_label
    sensor = sensor_options[selected_label]

    # --- Verificar si el sensor seleccionado es constante ---
    df_unit_filtered = df[df["unit_id"] == unidad]
    es_constante = False
    if sensor in df_unit_filtered.columns:
        valores = df_unit_filtered[sensor].dropna()
        # Un sensor es constante si tiene desviación estándar = 0 o todos los valores son iguales
        es_constante = (valores.std() == 0) or (valores.nunique() == 1)

    # --- Checkbox mostrar estacionaria (si existe pickle y el sensor NO es constante) ---
    show_stationary = False
    if stationary_map is not None:
        if es_constante:
            ui.checkbox("Mostrar serie estacionaria", value=False, disabled=True, 
                       help="Este sensor tiene valores constantes. No tiene sentido mostrar serie estacionaria.")
        else:
            show_stationary = ui.checkbox("Mostrar serie estacionaria", value=True)

    return {
        "unit_id": unidad,
        "sensor": sensor,
        "sensor_label": selected_label,
        "show_stationary": show_stationary
    }

#behavior
def controls_behavior(df: pd.DataFrame, location: str = "sidebar") -> dict:
    """
    Controles de la vista de comportamiento (ejemplo general).
    Devuelve un diccionario con los parámetros seleccionados.
    """
    ui = st.sidebar if location == "sidebar" else st

    # --- Selector de sensores (excluye configuraciones operacionales) ---
    sensores = helpers.detect_sensor_columns(df, exclude={'unit_id', 'time_cycles', 'RUL', 'rul', 'motor', 'ciclo', 
                                                           'config1', 'config2', 'config3',
                                                           'op_setting_1', 'op_setting_2', 'op_setting_3'})
    
    # Crear diccionario con nombres descriptivos para el selectbox
    from core import config
    sensor_options = {}
    for sensor in sensores:
        if sensor in config.SENSOR_SHORT_NAMES:
            # Usar solo el nombre corto del sensor
            label = config.SENSOR_SHORT_NAMES[sensor]
            sensor_options[label] = sensor
        else:
            sensor_options[sensor] = sensor
    
    # Inicializar el estado del sensor si no existe
    if 'selected_sensor_label' not in st.session_state:
        st.session_state.selected_sensor_label = list(sensor_options.keys())[0]
    
    # Obtener el índice del sensor guardado en session_state
    sensor_labels = list(sensor_options.keys())
    default_index = sensor_labels.index(st.session_state.selected_sensor_label) if st.session_state.selected_sensor_label in sensor_labels else 0
    
    # Mostrar selectbox con nombres descriptivos
    selected_label = ui.selectbox("Sensor", sensor_labels, index=default_index, key="behavior_sensor_selector")
    
    # Actualizar el estado global
    st.session_state.selected_sensor_label = selected_label
    sensor = sensor_options[selected_label]

    return {
        'sensor': sensor,
        'sensor_label': selected_label
    }


# KPIS y metricas
def kpis_unidad(df_unit: pd.DataFrame, unit_id: int | str):
    """
    Muestra un resumen textual de la unidad seleccionada,
    en formato compacto (estilo caption), similar a describe_filters().
    Incluye:
    - ID de unidad
    - Ciclo máximo
    - Número de sensores activos
    - Total de registros
    """
    if df_unit.empty:
        st.warning(f"No hay datos para la unidad {unit_id}.")
        return

    ciclo_max = int(df_unit["time_cycles"].max()) if "time_cycles" in df_unit.columns else "-"
    total_registros = len(df_unit)
    num_sensores = len(helpers.detect_sensor_columns(df_unit))

    texto = (
        f"**Unidad:** {unit_id} | "
        f"**Ciclo máximo:** {ciclo_max} | "
        f"**Sensores activos:** {num_sensores} | "
        f"**Registros:** {total_registros}"
    )

    st.caption(texto)
    st.divider()


# Tablas y resumenes
def show_lifetable(df: pd.DataFrame):
    """
    Renderiza la tabla de ciclos máximos por unidad.
    """
    tabla = helpers.build_lifetable(df)
    with st.expander("Ciclos máximos por unidad", expanded=False):
        st.dataframe(helpers.sanitize_dataframe_for_display(tabla), width="stretch")


def show_sensor_summary(df_unit: pd.DataFrame, sensor: str):
    """
    Muestra estadísticas descriptivas del sensor seleccionado.
    """
    stats = helpers.summarize_unit(df_unit, sensor)
    if not stats:
        st.info("No se pudieron calcular estadísticas para este sensor.")
        return

    st.markdown(f"#### Resumen - {sensor}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mínimo", f"{stats['min']:.3f}")
    c2.metric("Máximo", f"{stats['max']:.3f}")
    c3.metric("Promedio", f"{stats['mean']:.3f}")
    c4.metric("Desviación", f"{stats['std']:.3f}")
    c5.metric("Registros", stats['count'])

# Mensajes
def describe_filters(params: dict):
    """
    Muestra un resumen textual de los filtros activos.
    """
    activos = [f"{k}: {v}" for k, v in params.items() if v is not None]
    if activos:
        st.caption("**Filtros activos:** " + " | ".join(activos))


# Componentes reutilizables
def download_button(fig, filename="grafico.png"):
    """
    Muestra un botón para descargar una figura de Matplotlib,
    reiniciando temporalmente el estilo a su estado original.
    """
    import io
    import matplotlib.pyplot as plt

    # guardar estado actual (por si estás usando estilos personalizados)
    plt.rcdefaults()

    # exportar figura en estilo original
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300, facecolor="white")
    buf.seek(0)

    # mostrar botón de descarga
    st.download_button(
        label="Descargar imagen",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png",
    )

def download_button_data(df: pd.DataFrame, filename: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label='Descargar data como csv',
        data=csv,
        file_name=filename,
        mime="text/csv",
    )