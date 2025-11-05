# views/overview.py
# 
# VISTA PRINCIPAL - OVERVIEW DEL DASHBOARD
# 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core import data, config, helpers

class View:
    id = "overview"
    label = "Dashboard Principal"
    layout = "wide"

    @st.cache_data(show_spinner=False)
    def load_main_data(_self):
        """Carga el dataset principal"""
        df = data.load_data(config.PATHS['train_data'], ' ', header=None)
        df.drop(columns=[26, 27], inplace=True)
        df.columns = config.COLUMN_NAMES
        
        # Eliminar columnas constantes
        columnas_constantes = []
        for col in df.columns:
            if df[col].min() == df[col].max():
                columnas_constantes.append(col)
        df.drop(columns=columnas_constantes, inplace=True)
        
        return df

    def render(self):
        """Renderiza la vista principal del dashboard"""
        
        # Aplicar CSS personalizado
        st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)
        
        # 
        # HEADER
        # 
        
        col_title, col_logo = st.columns([3, 1])
        
        with col_title:
            st.title("Dashboard de Análisis para Mantenimiento Predictivo - FD001")
            st.markdown("""
                <div style='font-size: 1.1rem; color: #555555; margin-bottom: 2rem;'>
                    <b>Análisis de Supervivencia y Degradación de Motores Jet - NASA C-MAPSS FD001 Dataset</b><br>
                    <span style='color: #8D99AE;'>
                        Sistema de detección temprana de fallos basado en Machine Learning y análisis de series temporales
                    </span>
                </div>
            """, unsafe_allow_html=True)
        # 
        # CARGAR DATOS
        # 
        
        with st.spinner('Cargando datos...'):
            df = self.load_main_data()
        
        
        # 
        # RESUMEN DEL PROYECTO
        # 
        
        st.markdown("### Sobre este Proyecto")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="info-card">
                    <h3 style="color: white; margin-top: 0;"> Objetivo</h3>
                    <p>Desarrollar un sistema de <b>mantenimiento predictivo</b> que identifique 
                    fallos inminentes en motores jet antes de que ocurran, utilizando datos 
                    históricos de sensores y técnicas de ML.</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="success-card">
                    <h3 style="color: white; margin-top: 0;"> Metodología</h3>
                    <p>Análisis exploratorio de datos (EDA), preprocesamiento avanzado, 
                    selección de características, y modelado con <b>redes LSTM bidireccionales</b> 
                    para capturar patrones temporales de degradación.</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="warning-card">
                    <h3 style="color: white; margin-top: 0;">Impacto</h3>
                    <p>Reducción de costos operacionales, optimización de mantenimiento, 
                    prevención de fallos catastróficos y maximización de la <b>disponibilidad</b> 
                    de equipos críticos en industria aeroespacial.</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.divider()

        # 
        # KPIs PRINCIPALES
        # 
        
        st.markdown("### Indicadores Clave para Entrenamiento del Sistema")
        
        # Calcular KPIs
        total_motores = df['motor'].nunique()
        total_ciclos = len(df)
        ciclos_max = df.groupby('motor')['ciclo'].max()
        vida_util_promedio = ciclos_max.mean()
        vida_util_min = ciclos_max.min()
        vida_util_max = ciclos_max.max()
        # Contar solo sensores reales (excluir configuraciones operacionales)
        exclude_cols = {'motor', 'ciclo', 'unit_id', 'time_cycles', 'RUL', 'rul',
                       'config1', 'config2', 'config3', 
                       'op_setting_1', 'op_setting_2', 'op_setting_3'}
        total_sensores = len([col for col in df.columns if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)])
        
        # Mostrar KPIs en columnas
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        
        with kpi1:
            st.metric(
                label="Motores Analizados",
                value=f"{total_motores}",
                delta="Dataset FD001"
            )
        
        with kpi2:
            st.metric(
                label=" Ciclos Totales",
                value=f"{total_ciclos:,}",
                delta=f"{total_ciclos/total_motores:.0f} promedio/motor"
            )
        
        with kpi3:
            st.metric(
                label="Vida Útil Promedio",
                value=f"{vida_util_promedio:.0f}",
                delta=f"{vida_util_min}-{vida_util_max} rango"
            )
        
        with kpi4:
            st.metric(
                label="Sensores Útiles",
                value=f"{total_sensores}",
                delta="Con variación"
            )
        
        with kpi5:
            # Calcular tasa de completitud de datos
            completitud = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric(
                label="Calidad de Datos",
                value=f"{completitud:.1f}%",
                delta="Completitud"
            )
        
        st.divider()
        
        # 
        # DISTRIBUCIÓN DE VIDA ÚTIL
        # 
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("### Distribución de Vida Útil de los Motores")
            
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='none')
            ax.set_facecolor('none')
            
            # Obtener colores adaptativos
            from core.charts import get_theme_colors
            theme_colors = get_theme_colors()
            
            # Histograma
            counts, bins, patches = ax.hist(ciclos_max, bins=15, 
                                           color=config.COLORS['primary'], 
                                           alpha=0.7, edgecolor=theme_colors['edge'], linewidth=1.5)
            
            # Colorear barras según rangos
            for i, patch in enumerate(patches):
                bin_center = (bins[i] + bins[i+1]) / 2
                if bin_center < 150:
                    patch.set_facecolor(config.COLORS['danger'])
                elif bin_center < 250:
                    patch.set_facecolor(config.COLORS['warning'])
                else:
                    patch.set_facecolor(config.COLORS['success'])
            
            # Línea de promedio
            ax.axvline(vida_util_promedio, color=theme_colors['text'], 
                      linestyle='--', linewidth=2.5, label=f'Promedio: {vida_util_promedio:.0f} ciclos')
            
            # Estilo adaptativo
            ax.set_xlabel('Ciclos hasta el Fallo', fontweight='bold', fontsize=12, color=theme_colors['text'])
            ax.set_ylabel('Cantidad de Motores', fontweight='bold', fontsize=12, color=theme_colors['text'])
            ax.set_title('Distribución de Vida Útil - Motores Jet', 
                        fontweight='bold', fontsize=14, pad=15, color=theme_colors['title'])
            ax.grid(axis='y', alpha=0.3, linestyle='--', color=theme_colors['grid'])
            ax.tick_params(colors=theme_colors['text'], labelsize=10)
            
            # Leyenda con fondo adaptativo
            legend = ax.legend(fontsize=10, frameon=True, framealpha=0.9, 
                             facecolor=theme_colors['legend_bg'], edgecolor=theme_colors['legend_edge'])
            for text in legend.get_texts():
                text.set_color(theme_colors['legend_text'])
            
            # Agregar valores en barras
            for count, bin_edge, patch in zip(counts, bins, patches):
                if count > 0:
                    height = patch.get_height()
                    ax.text(patch.get_x() + patch.get_width()/2., height,
                           f'{int(count)}',
                           ha='center', va='bottom', fontweight='bold', fontsize=9, color=theme_colors['text'])
            
            plt.tight_layout()
            st.pyplot(fig, transparent=True)
        
        with col_right:
            st.markdown("### Estadísticas de Vida Útil")
            
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Mínimo</div>
                    <div class="kpi-value">{vida_util_min:.0f}</div>
                    <div class="kpi-subtitle">ciclos</div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-title">Promedio</div>
                    <div class="kpi-value">{vida_util_promedio:.0f}</div>
                    <div class="kpi-subtitle">ciclos</div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-title">Máximo</div>
                    <div class="kpi-value">{vida_util_max:.0f}</div>
                    <div class="kpi-subtitle">ciclos</div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-title">Desviación Estándar</div>
                    <div class="kpi-value">{ciclos_max.std():.0f}</div>
                    <div class="kpi-subtitle">ciclos</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Rangos de clasificación
            st.markdown("#### Clasificación por Rangos")
            baja = (ciclos_max < 150).sum()
            media = ((ciclos_max >= 150) & (ciclos_max < 250)).sum()
            alta = (ciclos_max >= 250).sum()
            
            st.markdown(f"""
                <span class="badge badge-danger">Vida Baja (&lt;150): {baja} motores</span><br>
                <span class="badge badge-warning">Vida Media (150-250): {media} motores</span><br>
                <span class="badge badge-success">Vida Alta (&gt;250): {alta} motores</span>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # 
        # MAPA DE CALOR - SENSORES
        # 
        
        st.markdown("### Mapa de Calor - Actividad de Sensores Claves")
        
        # Seleccionar solo sensores (excluir configuraciones operacionales)
        exclude_cols = {'motor', 'ciclo', 'unit_id', 'time_cycles', 'RUL', 'rul',
                       'config1', 'config2', 'config3',
                       'op_setting_1', 'op_setting_2', 'op_setting_3'}
        sensor_cols = [col for col in df.columns if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)]
        df_sensors = df[sensor_cols]
        
        # Calcular variabilidad (std) de cada sensor
        sensor_stats = df_sensors.std().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='none')
        ax.set_facecolor('none')
        
        # Crear heatmap de correlación entre sensores
        corr_matrix = df_sensors.corr()
        
        # Renombrar índices y columnas con nombres reales de sensores
        sensor_names_map = {col: config.SENSOR_SHORT_NAMES.get(col, col) for col in corr_matrix.columns}
        corr_matrix_renamed = corr_matrix.rename(index=sensor_names_map, columns=sensor_names_map)
        
        # Obtener colores adaptativos
        from core.charts import get_theme_colors
        theme_colors = get_theme_colors()
        
        sns.heatmap(corr_matrix_renamed, 
                   annot=False,
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'label': 'Correlación'},
                   ax=ax)
        
        # Estilos adaptativos
        ax.set_title('Matriz de Correlación entre Sensores', 
                    fontweight='bold', fontsize=14, pad=15, color=theme_colors['title'])
        
        # Color de las etiquetas de los ejes adaptativo
        ax.tick_params(colors=theme_colors['text'], labelsize=9)
        ax.set_xlabel('', color=theme_colors['text'])
        ax.set_ylabel('', color=theme_colors['text'])
        
        # Color de la barra de color (colorbar) adaptativo
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors=theme_colors['text'], labelsize=9)
        cbar.ax.yaxis.label.set_color(theme_colors['text'])
        
        # Rotar etiquetas para mejor legibilidad
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        
        # Mostrar sensores más variables
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Sensores Más Variables (Alta Información)")
            top_5 = sensor_stats.head(5)
            for idx, (sensor, value) in enumerate(top_5.items(), 1):
                sensor_name = config.SENSOR_SHORT_NAMES.get(sensor, sensor)
                st.markdown(f"""
                    <div class='sensor-box sensor-box-success'>
                        <b>{idx}. {sensor_name}</b> ({sensor}) - Desv. Est.: {value:.2f}
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Sensores Menos Variables (Baja Información)")
            bottom_5 = sensor_stats.tail(5).sort_values()
            for idx, (sensor, value) in enumerate(bottom_5.items(), 1):
                sensor_name = config.SENSOR_SHORT_NAMES.get(sensor, sensor)
                st.markdown(f"""
                    <div class='sensor-box sensor-box-warning'>
                        <b>{idx}. {sensor_name}</b> ({sensor}) - Desv. Est.: {value:.2f}
                    </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # 
        #  EQUIPO Y CRÉDITOS
        # 
        
        with st.expander(" Equipo de Desarrollo y Créditos"):
            st.markdown("""
                ### ‍ Autores
                
                - **Isaac David Sánchez Sánchez**
                - **Germán Eduardo de Armas Castaño**
                - **Katlyn Gutiérrez Cardona**
                - **Shalom Jhoanna Arrieta Marrugo**
                
                ---
                
                ###  Información Académica
                
                - **Curso:** Modelos de Regresión y Series de Tiempo con Aplicaciones en IA
                - **Universidad:** Universidad Tecnológica de Bolívar
                - **Grupo:** G, NCR 1705
                
                ---
                
                ### Dataset
                
                **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**
                - Subset: FD001 (Fault Dataset 001)
                - Fuente: NASA Prognostics Data Repository
                - Aplicación: Mantenimiento Predictivo en Motores a Reacción Turbofan de la NASA mediante Redes Neuronales Recurrentes LSTM
            """)
        
        # Footer
        st.markdown("""
            <div class="footer">
                <p><b>Dashboard de Mantenimiento Predictivo v1.0</b></p>
                <p>Universidad Tecnológica de Bolívar - 2025</p>
                <p style='font-size: 0.8rem; color: #8D99AE;'>
                    Desarrollado con  usando Streamlit, TensorFlow y Scikit-learn
                </p>
            </div>
        """, unsafe_allow_html=True)
