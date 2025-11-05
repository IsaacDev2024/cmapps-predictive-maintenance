# views/model.py
# 
# VISTA DE MODELO LSTM - PREDICCIONES Y EVALUACIÓN
# 

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from tensorflow import keras
from joblib import load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
from core import ui, config, helpers
import io

class View:
    id = "model"
    label = "Modelo LSTM - Predicciones"
    layout = "wide"

    @st.cache_resource(show_spinner=True)
    def load_model(_self):
        """Carga el modelo LSTM entrenado"""
        try:
            return keras.models.load_model('./data/modelo/modelo_lstm_completo.keras')
        except:
            return None

    @st.cache_data(show_spinner=False)
    def load_scaler(_self):
        """Carga el escalador utilizado en el preprocesamiento"""
        try:
            return load("./data/modelo/scaler_lstm.bin")
        except:
            return None

    @st.cache_data(show_spinner=False)
    def load_training_history(_self):
        """Carga el historial de entrenamiento"""
        try:
            with open("./data/modelo/historial_entrenamiento_lstm.pkl", "rb") as f:
                history = pickle.load(f)
            return history
        except:
            return None

    def style_axes(self, ax):
        """Aplica estilo coherente a los ejes con colores adaptativos"""
        from core.charts import get_theme_colors
        theme_colors = get_theme_colors()
        
        ax.tick_params(colors=theme_colors['text'], labelsize=10)
        ax.xaxis.label.set_color(theme_colors['text'])
        ax.yaxis.label.set_color(theme_colors['text'])
        for spine in ax.spines.values():
            spine.set_edgecolor(theme_colors['spine'])
            spine.set_alpha(0.6)
        ax.grid(alpha=0.35, color=theme_colors['grid'], linestyle="--")

    def load_user_data(self, uploaded_file):
        """
        Carga datos del usuario desde diferentes formatos
        Detecta automáticamente si la primera fila son headers o datos numéricos
        
        Formatos soportados:
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        - TXT (.txt) - separado por espacios o comas
        
        Returns:
            DataFrame con los datos cargados o None si hay error
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            df = None
            
            if file_extension == 'csv':
                # Intentar leer con header primero
                try:
                    df = pd.read_csv(uploaded_file)
                except:
                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, sep=';')
                    except:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, header=None)
                    
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
                
            elif file_extension == 'txt':
                # Intentar con espacio como separador (formato NASA)
                try:
                    df = pd.read_csv(uploaded_file, sep=r'\s+', engine='python')
                except:
                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, sep=',')
                    except:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, engine='python')
            else:
                st.error(f"Formato de archivo no soportado: {file_extension}")
                return None
            
            # DETECCIÓN AUTOMÁTICA DE HEADERS
            # Verificar si la primera fila son nombres de columnas (texto) o datos numéricos
            if df is not None:
                # Si las columnas ya son números (0, 1, 2...), significa que se cargó sin header
                if df.columns[0] == 0 or isinstance(df.columns[0], int):
                    # Ya está sin header, todo correcto
                    pass
                else:
                    # Tiene nombres de columnas, verificar si son válidos o si deberían ser datos
                    # Intentar convertir la primera columna a numérico
                    first_col_name = str(df.columns[0])
                    try:
                        # Si el nombre de la columna se puede convertir a número, probablemente no es un header
                        float(first_col_name)
                        # Es un número, la primera fila son datos, no headers
                        # Recargar el archivo sin header
                        uploaded_file.seek(0)
                        if file_extension == 'csv':
                            try:
                                df = pd.read_csv(uploaded_file, header=None)
                            except:
                                uploaded_file.seek(0)
                                df = pd.read_csv(uploaded_file, sep=';', header=None)
                        elif file_extension in ['xlsx', 'xls']:
                            df = pd.read_excel(uploaded_file, header=None)
                        elif file_extension == 'txt':
                            df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, engine='python')
                    except (ValueError, TypeError):
                        # No se puede convertir a número, es un header válido (texto)
                        pass
            
            return df
            
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")
            return None

    def validate_and_prepare_data(self, df, scaler):
        """
        Valida y prepara los datos del usuario para predicción.
        ACEPTA DATASET COMPLETO - La limpieza es interna.
        
        Args:
            df: DataFrame con los datos del motor (puede venir con todas las columnas)
            scaler: Escalador entrenado
            
        Returns:
            tuple: (datos_normalizados, info_validacion) o (None, error_msg)
        """
        
        # PASO 0: VALIDACIÓN DE NÚMERO DE COLUMNAS
        num_columnas = len(df.columns)
        
        if num_columnas > 26:
            # Verificar si las columnas extra (después de la 26) son todas nulas
            columnas_extra = df.iloc[:, 26:]
            if columnas_extra.isnull().all().all():
                st.warning(f"⚠️ Se detectaron {num_columnas - 26} columnas extra con valores nulos. Eliminando automáticamente...")
                df = df.iloc[:, :26]
                st.success(f"✓ Columnas nulas eliminadas. Ahora el dataset tiene {len(df.columns)} columnas.")
            else:
                return None, f"❌ Error: El dataset tiene {num_columnas} columnas. Se esperan exactamente 26 columnas (id + ciclo + 3 de configuración + 21 sensores).\n\nLas columnas extra contienen datos y no pueden ser eliminadas automáticamente."
        
        elif num_columnas < 26:
            return None, f"❌ Error: El dataset tiene solo {num_columnas} columnas. Se requieren exactamente 26 columnas (id + ciclo + 3 de configuración + 21 sensores)."
        
        # Si llegamos aquí, tenemos exactamente 26 columnas o se corrigió exitosamente
        
        # VALIDACIÓN DE VALORES NULOS EN LAS PRIMERAS 26 COLUMNAS
        valores_nulos = df.isnull().sum()
        columnas_con_nulos = valores_nulos[valores_nulos > 0]
        
        if len(columnas_con_nulos) > 0:
            # Crear mensaje detallado de columnas con nulos
            detalle_nulos = "\n".join([f"  - Columna {col}: {count} valores nulos" for col, count in columnas_con_nulos.items()])
            return None, f"❌ Error: El dataset contiene valores nulos que deben ser corregidos.\n\n{detalle_nulos}\n\nPor favor, proporciona un dataset con valores válidos en todas las celdas."
        
        # Nombres reales de los sensores NASA C-MAPSS
        # Formato: unit_id, time_cycles, op_setting_1, op_setting_2, op_setting_3, + 21 sensores
        nasa_sensor_names = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 
                            'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 
                            'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
        
        # Mapeo de sensores NASA a formato interno (sensor1-sensor21)
        sensor_mapping = {nasa: f'sensor{i+1}' for i, nasa in enumerate(nasa_sensor_names)}
        
        # Sensores seleccionados por correlación >= 0.2 (14 features)
        # Formato interno usado en el entrenamiento
        selected_sensors = ['sensor2', 'sensor3', 'sensor4', 'sensor7', 'sensor8', 
                           'sensor9', 'sensor11', 'sensor12', 'sensor13', 'sensor14', 
                           'sensor15', 'sensor17', 'sensor20', 'sensor21']
        
        # PASO 1: Detectar formato y normalizar columnas
        if df.columns[0] == 0 or isinstance(df.columns[0], int):
            # Formato sin headers (tipo NASA raw) - proceso silencioso
            # En este punto siempre tenemos exactamente 26 columnas gracias a la validación previa
            df.columns = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + nasa_sensor_names
        
        # PASO 2: Renombrar sensores NASA a formato interno si es necesario
        # Si el DataFrame tiene nombres NASA, convertirlos a sensor1, sensor2, etc. - proceso silencioso
        if any(col in nasa_sensor_names for col in df.columns):
            df = df.rename(columns=sensor_mapping)
        
        # VALIDACIÓN DE UNIT_ID - Debe ser el mismo en todas las filas
        if 'unit_id' in df.columns:
            unique_units = df['unit_id'].unique()
            if len(unique_units) > 1:
                return None, f"❌ Error: El dataset contiene datos de múltiples motores (unit_id).\n\nSe detectaron {len(unique_units)} motores diferentes: {list(unique_units)}\n\nPor favor, proporciona datos de un solo motor (todas las filas deben tener el mismo unit_id)."
            st.success(f"✓ Validación de motor: unit_id = {unique_units[0]} (único)")
        
        # VALIDACIÓN Y ORDENAMIENTO POR TIME_CYCLES
        if 'time_cycles' in df.columns:
            # Verificar si ya está ordenado
            if not df['time_cycles'].is_monotonic_increasing:
                st.warning("⚠️ Las filas no están ordenadas por time_cycles. Ordenando automáticamente...")
                df = df.sort_values(by='time_cycles', ascending=True).reset_index(drop=True)
                st.success("✓ Datos ordenados correctamente por time_cycles (ascendente)")
            else:
                st.success("✓ Datos ya están ordenados por time_cycles")
            
            # Verificar que los ciclos sean consecutivos o al menos crecientes
            ciclos = df['time_cycles'].values
            if len(ciclos) > 1:
                gaps = np.diff(ciclos)
                if np.any(gaps <= 0):
                    return None, f"❌ Error: Los valores de time_cycles deben ser estrictamente crecientes.\n\nSe detectaron ciclos duplicados o decrecientes."
        
        # PASO 3: Limpiar columnas no necesarias - proceso silencioso
        # Mantener time_cycles para el análisis posterior, solo eliminar configuraciones y unit_id
        cols_to_drop = ['op_setting_1', 'op_setting_2', 'op_setting_3', 
                       'config1', 'config2', 'config3', 
                       'unit_id', 'RUL', 'rul']
        
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(columns=existing_cols_to_drop)
        
        # PASO 4: Detectar columnas de sensores disponibles
        sensor_cols = [col for col in df.columns if 'sensor' in col.lower()]
        
        if len(sensor_cols) == 0:
            return None, "No se detectaron columnas de sensores. Asegúrate de que el archivo contenga datos de sensores."
        
        # PASO 4: Verificar que tenemos los sensores necesarios para el modelo
        missing_sensors = [s for s in selected_sensors if s not in sensor_cols]
        if missing_sensors:
            return None, f"Faltan sensores requeridos por el modelo: {', '.join(missing_sensors)}.\n\nEl modelo necesita estos 14 sensores específicos."
        
        # Información de validación
        info = {
            'total_ciclos': len(df),
            'sensores_detectados': len(sensor_cols),
            'sensores_usados': len(selected_sensors),
            'columnas_totales': list(df.columns),
            'sensores_disponibles': sensor_cols,
            'sensores_modelo': selected_sensors,
            'df_procesado': df  # Incluir DataFrame procesado con time_cycles
        }
        
        # PASO 5: Validar que haya suficientes ciclos (mínimo 30 para LSTM)
        if len(df) < 30:
            return None, f"Se necesitan al menos 30 ciclos para la predicción. El archivo tiene solo {len(df)} ciclos."
        
        # PASO 6: Extraer SOLO los 14 sensores seleccionados (en el orden correcto)
        try:
            # Mantener como DataFrame para preservar nombres de columnas
            sensor_data_df = df[selected_sensors]
            
            # Normalizar usando el scaler entrenado (que espera 14 features con nombres) - proceso silencioso
            sensor_data_normalized = scaler.transform(sensor_data_df)
            
            return sensor_data_normalized, info
            
        except Exception as e:
            return None, f"Error al procesar los datos: {str(e)}"

    def create_sequences(self, data, sequence_length=30):
        """
        Crea secuencias temporales para el modelo LSTM
        
        Args:
            data: Datos normalizados
            sequence_length: Longitud de la secuencia (debe ser 30)
            
        Returns:
            Array con secuencias listas para predicción
        """
        sequences = []
        
        # Tomar los últimos 30 ciclos (más recientes)
        if len(data) >= sequence_length:
            sequence = data[-sequence_length:]
            sequences.append(sequence)
        else:
            # Si hay menos de 30, hacer padding con ceros al inicio
            padding_needed = sequence_length - len(data)
            padded_sequence = np.vstack([np.zeros((padding_needed, data.shape[1])), data])
            sequences.append(padded_sequence)
        
        return np.array(sequences)

    def render(self):
        """Renderiza la vista principal del modelo LSTM"""
        
        # Aplicar CSS
        st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)
        
        # 
        # HEADER
        # 
        
        st.title("Sistema de Predicción LSTM - Detección de Fallos")
        st.markdown("""
            <div style='font-size: 1rem; color: #555555; margin-bottom: 2rem;'>
                Carga los datos de un motor que hizo sus ciclos sobre el nivel del mar y obtén predicciones en tiempo real sobre su estado de salud (HPC Degradation). </br>
                El modelo LSTM analiza las últimas 30 lecturas de sensores para determinar si el motor está en estado <b>NORMAL</b> o <b>FALLO INMINENTE</b>.
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # 
        #  CARGAR MODELO Y ARTEFACTOS
        # 
        
        modelo = self.load_model()
        scaler = self.load_scaler()
        history = self.load_training_history()
        
        if modelo is None or scaler is None:
            st.error("**Error:** No se pudo cargar el modelo LSTM o el escalador. Verifica que los archivos existan en `data/modelo/`")
            st.stop()
        
        # 
        # SECCIÓN 1: PREDICCIÓN INTERACTIVA
        # 
        
        st.markdown("##  Predicción en Tiempo Real")
        
        st.markdown("""
            <div style='padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 border-radius: 15px; color: white; margin-bottom: 2rem;'>
                <h3 style='color: white; margin-top: 0;'> Carga los Datos del Motor (Sobre el Nivel del Mar)</h3>
                <p style='margin-bottom: 0.5rem;'>
                    <b>IMPORTANTE:</b> El motor debe haber realizado sus ciclos sobre el nivel del mar.<br>
                    <b>Formatos aceptados:</b> CSV (.csv), Excel (.xlsx, .xls), TXT (.txt)s
                </p>
                <div style='margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.15); border-radius: 10px; border-left: 4px solid #FFD700;'>
                    <p style='margin: 0; font-size: 0.9rem;'>
                        <b>⚠️ IMPORTANTE - Orden de las Columnas:</b><br>
                        El dataset debe tener <b>exactamente 26 columnas</b> en el siguiente orden:<br>
                        1. unit_id | 2. time_cycles | 3-5. op_setting (3 columnas) | 6-26. sensores (21 columnas: T2, T24, T30, T50, P2, P15, P30, Nf, Nc, epr, Ps30, phi, NRf, NRc, BPR, farB, htBleed, Nf_dmd, PCNfR_dmd, W31, W32)<br>
                        ✓ Sin valores nulos o vacíos<br>
                        ✓ Mínimo 30 filas (ciclos) requeridas
                    </p>
                </div>
                <p style='font-size: 0.9rem; margin-top: 1rem; margin-bottom: 0;'>
                    <b>📖 Más detalles:</b> Expande "Ver Ejemplo de Formato de Datos" abajo para ver la estructura completa del dataset NASA C-MAPSS
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            " Selecciona el archivo con los datos del motor",
            type=['csv', 'xlsx', 'xls', 'txt'],
            help="Sube el dataset COMPLETO (raw) - El sistema automáticamente limpiará y preparará los datos"
        )
        
        if uploaded_file is not None:
            st.success(f"Archivo cargado: **{uploaded_file.name}**")
            
            # Cargar datos
            with st.spinner("Cargando y procesando datos..."):
                df_motor = self.load_user_data(uploaded_file)
            
            if df_motor is not None:
                
                # 
                # VISTA PREVIA DE DATOS
                # 
                
                with st.expander(" Vista Previa de los Datos Cargados", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(" Filas (Ciclos)", len(df_motor))
                    with col2:
                        st.metric(" Columnas", len(df_motor.columns))
                    with col3:
                        sensor_cols_count = len([col for col in df_motor.columns if 'sensor' in str(col).lower()])
                        st.metric("Sensores Detectados", sensor_cols_count)
                    with col4:
                        memoria = df_motor.memory_usage(deep=True).sum() / 1024
                        st.metric("Tamaño", f"{memoria:.2f} KB")
                    
                    st.dataframe(helpers.sanitize_dataframe_for_display(df_motor.head(10)), width='stretch')
                
                # 
                # VALIDAR Y PREPARAR DATOS
                # 
                
                with st.spinner(" Validando y preparando datos para predicción..."):
                    datos_normalizados, info = self.validate_and_prepare_data(df_motor, scaler)
                
                if datos_normalizados is None:
                    st.error(info)  # info contiene el mensaje de error
                    st.stop()
                
                # Mostrar información de validación
                st.markdown("### Validación Exitosa")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                        <div class="success-card">
                            <h4 style='color: white; margin-top: 0;'>Información del Motor</h4>
                            <p style='margin: 0;'>
                                <b>Ciclos totales:</b> {info['total_ciclos']}<br>
                                <b>Sensores detectados:</b> {info['sensores_detectados']}<br>
                                <b>Sensores usados:</b> {info['sensores_usados']} (seleccionados por correlación)<br>
                                <b>Últimos 30 ciclos:</b> Usados para predicción
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    sensores_usados = info.get('sensores_modelo', [])
                    # Convertir nombres de sensores a nombres reales
                    sensores_nombres_reales = [config.SENSOR_SHORT_NAMES.get(sensor, sensor) for sensor in sensores_usados]
                    st.markdown(f"""
                        <div class="info-card">
                            <h4 style='color: white; margin-top: 0;'>Sensores Analizados</h4>
                            <p style='margin: 0; font-size: 0.85rem;'>
                                {', '.join(sensores_nombres_reales)}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # 
                #  REALIZAR PREDICCIÓN
                # 
                
                st.markdown("## Resultado de la Predicción")
                
                # Crear secuencias
                secuencias = self.create_sequences(datos_normalizados, sequence_length=30)
                
                # Realizar predicción
                with st.spinner("El modelo está analizando los datos..."):
                    probabilidad_fallo = modelo.predict(secuencias, verbose=0)[0][0]
                
                # Determinar estado
                umbral = 0.5
                estado = "FALLO INMINENTE" if probabilidad_fallo > umbral else "NORMAL"
                confianza = probabilidad_fallo if probabilidad_fallo > umbral else (1 - probabilidad_fallo)
                
                # 
                # MOSTRAR RESULTADO
                # 
                
                # Card de resultado
                if estado == "NORMAL":
                    st.markdown(f"""
                        <div style='padding: 3rem; background: linear-gradient(135deg, #06A77D 0%, #04E762 100%); 
                             border-radius: 20px; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.2);'>
                            <div style='font-size: 5rem; margin-bottom: 1rem;'></div>
                            <h2 style='color: white; margin: 0; font-size: 2.5rem;'>{estado}</h2>
                            <p style='color: white; font-size: 1.2rem; margin-top: 1rem;'>
                                El motor está operando dentro de parámetros normales
                            </p>
                            <div style='margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.2); border-radius: 10px;'>
                                <p style='color: white; font-size: 1rem; margin: 0;'>
                                    <b>Confianza:</b> {confianza*100:.1f}%
                                </p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style='padding: 3rem; background: linear-gradient(135deg, #D62828 0%, #F77F00 100%); 
                             border-radius: 20px; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.2);'>
                            <div style='font-size: 5rem; margin-bottom: 1rem;'></div>
                            <h2 style='color: white; margin: 0; font-size: 2.5rem;'>{estado}</h2>
                            <p style='color: white; font-size: 1.2rem; margin-top: 1rem;'>
                                Se detectaron patrones de degradación. Se recomienda mantenimiento preventivo.
                            </p>
                            <div style='margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.2); border-radius: 10px;'>
                                <p style='color: white; font-size: 1rem; margin: 0;'>
                                    <b>Confianza:</b> {confianza*100:.1f}%
                                </p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # 
                # ANÁLISIS DETALLADO
                # 
                
                st.markdown("## Análisis Detallado de Sensores")
                
                # Usar el DataFrame procesado que incluye time_cycles
                df_procesado = info.get('df_procesado')
                
                # Mostrar evolución de los últimos ciclos
                ultimos_ciclos = min(30, len(df_procesado))
                df_ultimos = df_procesado.tail(ultimos_ciclos)
                
                # Seleccionar sensor para visualizar - solo sensores usados por el modelo
                selected_sensors = info.get('sensores_modelo', [])
                
                if len(selected_sensors) > 0:
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        # Crear opciones con nombres reales
                        sensor_options = {
                            f"{config.SENSOR_SHORT_NAMES.get(s, s)} ({s})": s 
                            for s in selected_sensors
                        }
                        
                        sensor_display = st.selectbox(
                            "Selecciona un sensor para analizar:",
                            list(sensor_options.keys()),
                            key="sensor_analisis"
                        )
                        
                        # Obtener el nombre técnico del sensor seleccionado
                        sensor_seleccionado = sensor_options[sensor_display]
                    
                    with col2:
                        # Gráfico de evolución
                        fig_evol, ax = plt.subplots(figsize=(14, 5), facecolor='none')
                        ax.set_facecolor('none')
                        
                        # Obtener colores adaptativos
                        from core.charts import get_theme_colors
                        theme_colors = get_theme_colors()
                        
                        # Usar time_cycles si está disponible, si no usar índice
                        if 'time_cycles' in df_ultimos.columns:
                            ciclos = df_ultimos['time_cycles'].values
                            xlabel = 'Ciclos (Time Cycles)'
                        else:
                            ciclos = list(range(len(df_ultimos)))
                            xlabel = 'Ciclos (Últimos 30)'
                        
                        valores = df_ultimos[sensor_seleccionado].values
                        
                        # Obtener nombre real del sensor
                        sensor_nombre_real = config.SENSOR_SHORT_NAMES.get(sensor_seleccionado, sensor_seleccionado)
                        
                        ax.plot(ciclos, valores, 
                               color=config.COLORS['primary'], 
                               linewidth=2.5, marker='o', markersize=5,
                               label=f'{sensor_nombre_real} ({sensor_seleccionado})')
                        
                        # Línea de tendencia
                        z = np.polyfit(range(len(ciclos)), valores, 2)
                        p = np.poly1d(z)
                        ax.plot(ciclos, p(range(len(ciclos))), 
                               color=config.COLORS['danger'], 
                               linestyle='--', linewidth=2, alpha=0.7,
                               label='Tendencia')
                        
                        # Estilo adaptativo
                        ax.set_xlabel(xlabel, fontweight='bold', fontsize=12, color=theme_colors['text'])
                        ax.set_ylabel('Valor del Sensor', fontweight='bold', fontsize=12, color=theme_colors['text'])
                        ax.set_title(f'Evolución de {sensor_nombre_real} - Últimos Ciclos', 
                                    fontweight='bold', fontsize=14, pad=15, color=theme_colors['title'])
                        ax.tick_params(colors=theme_colors['text'], labelsize=10)
                        ax.grid(alpha=0.3, linestyle='--', color=theme_colors['grid'])
                        
                        # Leyenda con fondo adaptativo
                        legend = ax.legend(frameon=True, framealpha=0.9, 
                                         facecolor=theme_colors['legend_bg'], edgecolor=theme_colors['legend_edge'])
                        for text in legend.get_texts():
                            text.set_color(theme_colors['legend_text'])
                        
                        plt.tight_layout()
                        st.pyplot(fig_evol, transparent=True)
                
                # Recomendaciones
                st.markdown("### Recomendaciones")
                
                if estado == "NORMAL":
                    st.markdown("""
                        <div style='padding: 1.5rem; background: #06A77D22; border-left: 4px solid #06A77D; border-radius: 8px;'>
                            <h4 style='color: #06A77D; margin-top: 0;'>Motor en Estado Óptimo</h4>
                            <ul style='margin-bottom: 0;'>
                                <li>Continuar con el programa de mantenimiento regular</li>
                                <li>Monitorear sensores clave periódicamente</li>
                                <li>Mantener registro de lecturas para análisis histórico</li>
                                <li>Próxima revisión según calendario establecido</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style='padding: 1.5rem; background: #D6282822; border-left: 4px solid #D62828; border-radius: 8px;'>
                            <h4 style='color: #D62828; margin-top: 0;'>Acción Inmediata Requerida</h4>
                            <ul style='margin-bottom: 0;'>
                                <li><b>Programar inspección técnica urgente</b></li>
                                <li>Revisar historial de mantenimiento reciente</li>
                                <li>Analizar sensores con mayor variación</li>
                                <li>Considerar reducción de carga operativa</li>
                                <li>Preparar componentes de reemplazo críticos</li>
                                <li>Documentar todas las observaciones</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
        
        else:
            
            # Ejemplo de formato
            with st.expander(" Ver Ejemplo de Formato de Datos", expanded=False):
                st.markdown("###  Formato Esperado - NASA C-MAPSS Dataset")
                
                ejemplo_data = pd.DataFrame({
                    'unit_id': [1, 1, 1, 1, 1],
                    'time_cycles': [1, 2, 3, 4, 5],
                    'op_setting_1': [-0.0007, -0.0004, -0.0003, -0.0007, -0.0019],
                    'op_setting_2': [-0.0004, -0.0002, -0.0001, -0.0008, -0.0021],
                    'op_setting_3': [100.0, 100.0, 100.0, 100.0, 100.0],
                    'T2': [518.67, 518.67, 518.67, 518.67, 518.67],
                    'T24': [641.82, 642.15, 642.35, 642.35, 642.37],
                    'T30': [1589.70, 1591.82, 1587.99, 1582.79, 1582.85],
                    'T50': [1400.60, 1403.14, 1404.20, 1401.87, 1406.22],
                    '...': ['...', '...', '...', '...', '...']
                })
                
                st.dataframe(helpers.sanitize_dataframe_for_display(ejemplo_data), width='stretch', hide_index=True)
        
        st.divider()
        
        # 
        # SECCIÓN 2: INFORMACIÓN DEL MODELO
        # 
        
        with st.expander("Información del Modelo LSTM", expanded=False):
            st.markdown("### Arquitectura del Modelo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    **Tipo:** Bidirectional LSTM  
                    **Capas LSTM:** 64 (bidireccional) + 32  
                    **Capas Densas:** 32 + 1 (sigmoid)  
                    **Dropout:** 0.3, 0.3, 0.2  
                    **Regularización:** L2 (0.01)  
                """)
            
            with col2:
                st.markdown("""
                    **Longitud de Secuencia:** 30 ciclos  
                    **Optimizador:** Adam (lr=0.001)  
                    **Función de Pérdida:** Binary Crossentropy  
                    **Métricas:** Accuracy, Precision, Recall  
                """)
            
            # Historial de entrenamiento
            if history is not None:
                st.markdown("### Historial de Entrenamiento")
                
                # Obtener colores adaptativos
                from core.charts import get_theme_colors
                theme_colors = get_theme_colors()
                
                fig_hist, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='none')
                for ax in axes:
                    ax.set_facecolor('none')
                
                # Pérdida
                axes[0].plot(history['loss'], label='Entrenamiento', 
                           color=config.COLORS['primary'], linewidth=2)
                axes[0].plot(history['val_loss'], label='Validación', 
                           color=config.COLORS['danger'], linewidth=2)
                axes[0].set_xlabel('Épocas', fontweight='bold', color=theme_colors['text'])
                axes[0].set_ylabel('Pérdida', fontweight='bold', color=theme_colors['text'])
                axes[0].set_title('Evolución de la Pérdida', fontweight='bold', color=theme_colors['title'])
                axes[0].tick_params(colors=theme_colors['text'], labelsize=10)
                axes[0].grid(alpha=0.3, color=theme_colors['grid'])
                legend0 = axes[0].legend(frameon=True, framealpha=0.9, 
                                        facecolor=theme_colors['legend_bg'], edgecolor=theme_colors['legend_edge'])
                for text in legend0.get_texts():
                    text.set_color(theme_colors['legend_text'])
                
                # Precisión
                if 'accuracy' in history:
                    axes[1].plot(history['accuracy'], label='Entrenamiento', 
                               color=config.COLORS['primary'], linewidth=2)
                    axes[1].plot(history['val_accuracy'], label='Validación', 
                               color=config.COLORS['danger'], linewidth=2)
                    axes[1].set_xlabel('Épocas', fontweight='bold', color=theme_colors['text'])
                    axes[1].set_ylabel('Precisión', fontweight='bold', color=theme_colors['text'])
                    axes[1].set_title('Evolución de la Precisión', fontweight='bold', color=theme_colors['title'])
                    axes[1].tick_params(colors=theme_colors['text'], labelsize=10)
                    axes[1].grid(alpha=0.3, color=theme_colors['grid'])
                    legend1 = axes[1].legend(frameon=True, framealpha=0.9, 
                                            facecolor=theme_colors['legend_bg'], edgecolor=theme_colors['legend_edge'])
                    for text in legend1.get_texts():
                        text.set_color(theme_colors['legend_text'])
                
                plt.tight_layout()
                st.pyplot(fig_hist, transparent=True)
        
        # Footer
        st.markdown("""
            <div class="footer">
                <p><b>Sistema de Predicción LSTM v1.0</b></p>
                <p>Modelo entrenado con 100 motores del dataset NASA C-MAPSS FD001</p>
                <p style='font-size: 0.8rem; color: #8D99AE;'>
                    Desarrollado con TensorFlow/Keras - Universidad Tecnológica de Bolívar
                </p>
            </div>
        """, unsafe_allow_html=True)
