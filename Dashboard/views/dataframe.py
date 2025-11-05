# views/dataframe.py
# 
# VISTA DE EXPLORADOR DE DATOS
# 

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from core import ui, helpers, charts, data, config

class View:
    id = "dataframe"
    label = "Explorador de Datos"
    layout = "sidebar"
    
    @st.cache_data(show_spinner=False)
    def load_FD001_data(_self):
        """Carga y normaliza el dataset principal de motores NASA C-MAPSS FD001."""
        df = data.load_data(config.PATHS['train_data'], ' ', header=None)
        df.drop(columns=[26, 27], inplace=True)
        df.columns = config.COLUMN_NAMES
        return df

    def render(self):
        """Muestra explorador interactivo de datos."""
        
        # Aplicar CSS
        st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)
        
        # 
        # HEADER
        # 
        
        st.title("Explorador Interactivo de Datos - FD001")
        st.markdown("""
            <div style='font-size: 1rem; color: #555555; margin-bottom: 2rem;'>
                Explora, filtra y analiza los datasets de entrenamiento. 
                Visualiza estadísticas descriptivas, distribuciones y exporta datos filtrados.
            </div>
        """, unsafe_allow_html=True)

        # 
        # CARGAR DATOS
        # 
        
        df_FD = self.load_FD001_data()
        
        # 
        # DATASET AUTOMÁTICO: FD001
        # 
        
        # Cargar automáticamente el Dataset de Motores
        df = df_FD.copy()
        dataset_name = "FD001 - Motores"
        dataset_desc = "Dataset completo de sensores de motores jet NASA C-MAPSS"
        
        st.divider()
        
        # 
        # INFORMACIÓN GENERAL DEL DATASET
        # 
        
        st.markdown(f"### Información General - {dataset_name}")
        
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        
        with kpi1:
            st.metric(
                label=" Filas",
                value=f"{len(df):,}",
                delta=None
            )
        
        with kpi2:
            st.metric(
                label=" Columnas",
                value=f"{len(df.columns)}",
                delta=None
            )
        
        with kpi3:
            memoria_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric(
                label="Tamaño",
                value=f"{memoria_mb:.2f} MB",
                delta=None
            )
        
        with kpi4:
            if 'motor' in df.columns:
                motores = df['motor'].nunique()
            elif 'unit_id' in df.columns:
                motores = df['unit_id'].nunique()
            else:
                motores = "N/A"
            st.metric(
                label="Motores",
                value=f"{motores}",
                delta=None
            )
        
        with kpi5:
            valores_nulos = df.isnull().sum().sum()
            st.metric(
                label="Valores Nulos",
                value=f"{valores_nulos}",
                delta="Limpio" if valores_nulos == 0 else "Requiere limpieza"
            )
        
        st.divider()
        
        # 
        # FILTROS INTERACTIVOS
        # 
        
        st.markdown("### Filtros de Datos")
        
        with st.expander("Configurar Filtros", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Filtro por motor
                if 'motor' in df.columns:
                    motor_col = 'motor'
                elif 'unit_id' in df.columns:
                    motor_col = 'unit_id'
                else:
                    motor_col = None
                
                if motor_col:
                    motor_options = ['Todos'] + sorted(df[motor_col].unique().tolist())
                    selected_motor = st.selectbox(f"Filtrar por {motor_col}", motor_options)
                    
                    if selected_motor != 'Todos':
                        df = df[df[motor_col] == selected_motor]
            
            with col2:
                # Filtro por rango de filas
                max_rows = len(df)
                num_rows = st.slider("Número de filas a mostrar", 10, min(max_rows, 1000), min(100, max_rows))
            
            with col3:
                # Filtro por columnas
                all_columns = df.columns.tolist()
                
                # Crear mapeo de nombres visuales a nombres reales de columnas
                column_display_map = {}
                column_reverse_map = {}
                for col in all_columns:
                    if col in config.SENSOR_SHORT_NAMES:
                        display_name = f"{config.SENSOR_SHORT_NAMES[col]} ({col})"
                    else:
                        display_name = col
                    column_display_map[col] = display_name
                    column_reverse_map[display_name] = col
                
                # Mostrar nombres visuales en el multiselect
                display_columns = [column_display_map[col] for col in all_columns]
                default_display = [column_display_map[col] for col in (all_columns[:10] if len(all_columns) > 10 else all_columns)]
                
                selected_display_columns = st.multiselect(
                    "Seleccionar columnas",
                    display_columns,
                    default=default_display
                )
                
                # Convertir de vuelta a nombres reales de columnas
                selected_columns = [column_reverse_map[disp_col] for disp_col in selected_display_columns]
        
        # Aplicar filtros de columnas
        if selected_columns:
            df_filtered = df[selected_columns].head(num_rows)
        else:
            df_filtered = df.head(num_rows)
        
        st.divider()
        
        # 
        # VISTA DE DATOS
        # 
        
        st.markdown(f"### Vista de Datos ({len(df_filtered)} filas)")
        
        # Renombrar columnas para mostrar nombres reales de sensores
        df_display = df_filtered.copy()
        df_display.columns = [
            f"{config.SENSOR_SHORT_NAMES.get(col, col)} ({col})" if col in config.SENSOR_SHORT_NAMES else col 
            for col in df_display.columns
        ]
        
        st.dataframe(
            helpers.sanitize_dataframe_for_display(df_display),
            width='stretch',
            height=400
        )
        
        # Botón de descarga
        ui.download_button_data(df_filtered, f"{dataset_name.lower().replace(' ', '_')}_filtered.csv")
        
        st.divider()
        
        # 
        # ESTADÍSTICAS DESCRIPTIVAS
        # 
        
        st.markdown("### Estadísticas Descriptivas")
        
        tab1, tab2, tab3 = st.tabs(["Resumen Numérico", " Info del Dataset", " Tipos de Datos"])
        
        with tab1:
            st.markdown("#### Estadísticas de Variables Numéricas")
            stats_df = df.describe().transpose()
            stats_df['missing'] = df.isnull().sum()
            stats_df['missing %'] = (df.isnull().sum() / len(df) * 100).round(2)
            
            # Renombrar índice con nombres reales de sensores
            stats_df_display = stats_df.copy()
            stats_df_display.index = [
                f"{config.SENSOR_SHORT_NAMES.get(idx, idx)} ({idx})" if idx in config.SENSOR_SHORT_NAMES else idx 
                for idx in stats_df_display.index
            ]
            
            st.dataframe(
                helpers.sanitize_dataframe_for_display(stats_df_display).style.format({
                    'count': '{:.0f}',
                    'mean': '{:.3f}',
                    'std': '{:.3f}',
                    'min': '{:.3f}',
                    '25%': '{:.3f}',
                    '50%': '{:.3f}',
                    '75%': '{:.3f}',
                    'max': '{:.3f}',
                    'missing': '{:.0f}',
                    'missing %': '{:.2f}%'
                }).background_gradient(subset=['mean', 'std'], cmap='YlOrRd'),
                width='stretch'
            )
            
            # Botón de descarga
            ui.download_button_data(stats_df.reset_index(), f"{dataset_name.lower().replace(' ', '_')}_statistics.csv")
        
        with tab2:
            st.markdown("#### Información del Dataset")
            
            info_data = []
            for col in df.columns:
                # Obtener nombre real del sensor si existe
                col_display = f"{config.SENSOR_SHORT_NAMES.get(col, col)} ({col})" if col in config.SENSOR_SHORT_NAMES else col
                info_data.append({
                    'Columna': col_display,
                    'Tipo': str(df[col].dtype),
                    'No Nulos': df[col].count(),
                    'Nulos': df[col].isnull().sum(),
                    '% Nulos': f"{(df[col].isnull().sum() / len(df) * 100):.2f}%",
                    'Únicos': df[col].nunique(),
                    'Memoria (KB)': f"{df[col].memory_usage(deep=True) / 1024:.2f}"
                })
            
            info_df = pd.DataFrame(info_data)
            st.dataframe(helpers.sanitize_dataframe_for_display(info_df), width='stretch', hide_index=True)
        
        with tab3:
            st.markdown("#### Distribución de Tipos de Datos")
            
            type_counts = df.dtypes.value_counts()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                for dtype, count in type_counts.items():
                    st.markdown(f"""
                        <div style='padding: 0.75rem; margin: 0.5rem 0; background: #EDF2F4; border-radius: 8px; border-left: 4px solid {config.COLORS['primary']}'>
                            <b>{dtype}:</b> {count} columnas
                        </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                fig_types, ax = plt.subplots(figsize=(10, 6))
                
                colors_palette = [config.COLORS['primary'], config.COLORS['success'], 
                                config.COLORS['warning'], config.COLORS['danger']]
                
                wedges, texts, autotexts = ax.pie(
                    type_counts.values,
                    labels=type_counts.index,
                    autopct='%1.1f%%',
                    colors=colors_palette[:len(type_counts)],
                    startangle=90,
                    explode=[0.05] * len(type_counts)
                )
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(11)
                
                ax.set_title('Distribución de Tipos de Datos', fontweight='bold', fontsize=14, pad=15)
                
                plt.tight_layout()
                st.pyplot(fig_types, transparent=True)
        
        st.divider()
    
