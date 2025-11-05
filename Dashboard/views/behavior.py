# views/behavior.py
# 
# VISTA DE COMPORTAMIENTO - ANÁLISIS POR MOTOR
# 

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from core import ui, helpers, charts, data, config

class View:
    id = "behavior"
    label = "Comportamiento por Motor"
    layout = "sidebar"
    
    @st.cache_data(show_spinner=False)
    def load_main_data(_self):
        """Carga y normaliza el dataset principal."""
       
        df = data.load_data(config.PATHS['train_data'], ' ', header=None)
        df.drop(columns=[26, 27], inplace=True)
        df.columns = config.COLUMN_NAMES

        return df

    def render(self):
        """Muestra análisis del comportamiento de los motores"""
        
        # Aplicar CSS
        st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)
        
        # 
        # HEADER
        # 
        
        st.title("Análisis de Comportamiento por Motor - FD001")
        st.markdown("""
            <div style='font-size: 1rem; color: #555555; margin-bottom: 2rem;'>
                Explora los patrones de degradación de sensores a través de todos los motores.
                Identifica tendencias globales y compara el comportamiento entre diferentes unidades.
                Puede seleccionar los diferentes sensores en la parte inferior del Menú de Navegación.
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()

        # 
        # CARGAR DATOS
        # 
        
        df = self.load_main_data()

        # 
        # CONTROLES INTERACTIVOS
        # 
        
        params = ui.controls_behavior(df, location=self.layout)
        sensor = params.get('sensor')
        sensor_label = params.get('sensor_label', sensor)
        
        # 
        # KPIs GENERALES
        # 
        
        st.markdown("### Resumen General del Sensor")
        
        total_motores = df['motor'].nunique()
        total_lecturas = len(df)
        valor_min_global = df[sensor].min()
        valor_max_global = df[sensor].max()
        promedio_global = df[sensor].mean()
        std_global = df[sensor].std()
        
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        
        with kpi1:
            st.metric(
                label="Motores Totales",
                value=f"{total_motores}",
                delta=None
            )
        
        with kpi2:
            st.metric(
                label="Sensor Analizado",
                value=sensor_label,
                delta=f"{total_lecturas:,} lecturas"
            )
        
        with kpi3:
            st.metric(
                label="Rango Global",
                value=f"{valor_max_global:.2f}",
                delta=f"Min: {valor_min_global:.2f}"
            )
        
        with kpi4:
            st.metric(
                label="Promedio Global",
                value=f"{promedio_global:.3f}",
                delta=f"±{std_global:.3f}"
            )
        
        with kpi5:
            coef_var = (std_global / promedio_global * 100) if promedio_global != 0 else 0
            st.metric(
                label="Coef. Variación",
                value=f"{coef_var:.1f}%",
                delta="Variabilidad"
            )
        
        st.divider()
        
        # 
        # GRÁFICO PRINCIPAL: COMPORTAMIENTO POR MOTOR
        # 
        
        st.markdown(f"### Evolución del Sensor **{sensor_label}** - Todos los Motores")
        
        # Calcular estadísticas del sensor para insights
        sensor_mean = df[sensor].mean()
        sensor_std = df[sensor].std()
        sensor_min = df[sensor].min()
        sensor_max = df[sensor].max()
        
        st.markdown(f"""
            <div class='info-box-insight'>
                <div style='margin-bottom: 0.5rem;'>
                    <b>Interpretación:</b> Cada línea representa la evolución del <b>{sensor_label}</b> en un motor individual. 
                    La línea roja discontinua muestrea la tendencia global del sensor.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        fig = charts.make_behavior_plot(df, sensor, sensor_label=sensor_label)
        charts.show_chart(fig)
        ui.download_button(fig, filename=f"comportamiento_{sensor}_todos_motores.png")
        
        st.divider()
        
        # 
        # ANÁLISIS DE VIDA ÚTIL
        # 
        
        st.markdown("### Análisis de Vida Útil de los Motores")
        
        col_chart, col_stats = st.columns([2, 1])
        
        with col_chart:
            # Calcular ciclos máximos por motor
            ciclos_max = df.groupby('motor')['ciclo'].max().sort_values(ascending=False)
            
            fig_vida, ax = plt.subplots(figsize=(12, 8), facecolor='none')
            ax.set_facecolor('none')
            
            # Obtener colores adaptativos
            from core.charts import get_theme_colors
            theme_colors = get_theme_colors()
            
            # Gráfico de barras horizontales con colores según vida útil
            colors = []
            for ciclos in ciclos_max:
                if ciclos < 150:
                    colors.append(config.COLORS['danger'])
                elif ciclos < 250:
                    colors.append(config.COLORS['warning'])
                else:
                    colors.append(config.COLORS['success'])
            
            bars = ax.barh(range(len(ciclos_max)), ciclos_max.values, color=colors, 
                          alpha=0.7, edgecolor=theme_colors['edge'], linewidth=0.5)
            
            # Configuración adaptativa
            ax.set_yticks(range(len(ciclos_max)))
            ax.set_yticklabels([f'Motor {i}' for i in ciclos_max.index], fontsize=7)
            ax.set_xlabel('Ciclos hasta el Fallo', fontweight='bold', fontsize=12, color=theme_colors['text'])
            ax.set_ylabel('Motor', fontweight='bold', fontsize=12, color=theme_colors['text'])
            ax.set_title('Vida Útil de Cada Motor (Ordenado Descendente)', 
                        fontweight='bold', fontsize=14, pad=15, color=theme_colors['title'])
            ax.tick_params(colors=theme_colors['text'], labelsize=10)
            ax.grid(axis='x', alpha=0.3, linestyle='--', color=theme_colors['grid'])
            
            # Línea de promedio
            promedio_ciclos = ciclos_max.mean()
            ax.axvline(promedio_ciclos, color=theme_colors['text'], 
                      linestyle='--', linewidth=2.5, label=f'Promedio: {promedio_ciclos:.0f}')
            
            # Leyenda con fondo adaptativo
            legend = ax.legend(frameon=True, framealpha=0.9, 
                             facecolor=theme_colors['legend_bg'], edgecolor=theme_colors['legend_edge'])
            for text in legend.get_texts():
                text.set_color(theme_colors['legend_text'])
            
            plt.tight_layout()
            st.pyplot(fig_vida, transparent=True)
        
        with col_stats:
            st.markdown("#### Estadísticas de Vida Útil")
            
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Vida Mínima</div>
                    <div class="kpi-value">{ciclos_max.min():.0f}</div>
                    <div class="kpi-subtitle">ciclos</div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-title">Vida Máxima</div>
                    <div class="kpi-value">{ciclos_max.max():.0f}</div>
                    <div class="kpi-subtitle">ciclos</div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-title">Promedio</div>
                    <div class="kpi-value">{ciclos_max.mean():.0f}</div>
                    <div class="kpi-subtitle">ciclos</div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-title">Mediana</div>
                    <div class="kpi-value">{ciclos_max.median():.0f}</div>
                    <div class="kpi-subtitle">ciclos</div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-title">Desv. Estándar</div>
                    <div class="kpi-value">{ciclos_max.std():.0f}</div>
                    <div class="kpi-subtitle">ciclos</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Clasificación
            st.markdown("#### Clasificación")
            vida_baja = (ciclos_max < 150).sum()
            vida_media = ((ciclos_max >= 150) & (ciclos_max < 250)).sum()
            vida_alta = (ciclos_max >= 250).sum()
            
            st.markdown(f"""
                <span class="badge badge-danger">Baja (&lt;150): {vida_baja}</span><br>
                <span class="badge badge-warning">Media (150-250): {vida_media}</span><br>
                <span class="badge badge-success">Alta (&gt;250): {vida_alta}</span>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # 
        # COMPARACIÓN ENTRE MOTORES
        # 
        
        st.markdown("### Comparación Detallada entre Motores")
        
        # Permitir selección de motores para comparar
        col1, col2 = st.columns(2)
        
        with col1:
            motores_disponibles = sorted(df['motor'].unique())
            motor_1 = st.selectbox("Selecciona Motor 1", motores_disponibles, index=0, key="motor1")
        
        with col2:
            motor_2 = st.selectbox("Selecciona Motor 2", motores_disponibles, index=min(1, len(motores_disponibles)-1), key="motor2")
        
        if motor_1 != motor_2:
            # Filtrar datos de ambos motores
            df_motor1 = df[df['motor'] == motor_1]
            df_motor2 = df[df['motor'] == motor_2]
            
            # Gráfico comparativo
            fig_comp, ax = plt.subplots(figsize=(14, 6), facecolor='none')
            ax.set_facecolor('none')
            
            # Obtener colores adaptativos
            from core.charts import get_theme_colors
            theme_colors = get_theme_colors()
            
            ax.plot(df_motor1['ciclo'], df_motor1[sensor], 
                   color=config.COLORS['primary'], linewidth=2.5, alpha=0.8, 
                   label=f'Motor {motor_1} ({len(df_motor1)} ciclos)', marker='o', markersize=3)
            
            ax.plot(df_motor2['ciclo'], df_motor2[sensor], 
                   color=config.COLORS['danger'], linewidth=2.5, alpha=0.8, 
                   label=f'Motor {motor_2} ({len(df_motor2)} ciclos)', marker='s', markersize=3)
            
            # Estilo adaptativo
            ax.set_xlabel('Ciclo', fontweight='bold', fontsize=12, color=theme_colors['text'])
            ax.set_ylabel(sensor_label, fontweight='bold', fontsize=12, color=theme_colors['text'])
            ax.set_title(f'Comparación: Motor {motor_1} vs Motor {motor_2} - {sensor_label}', 
                        fontweight='bold', fontsize=14, pad=15, color=theme_colors['title'])
            ax.tick_params(colors=theme_colors['text'], labelsize=10)
            ax.grid(alpha=0.3, linestyle='--', color=theme_colors['grid'])
            
            # Leyenda con fondo adaptativo
            legend = ax.legend(fontsize=11, frameon=True, framealpha=0.9, 
                             facecolor=theme_colors['legend_bg'], edgecolor=theme_colors['legend_edge'])
            for text in legend.get_texts():
                text.set_color(theme_colors['legend_text'])
            
            plt.tight_layout()
            st.pyplot(fig_comp, transparent=True)
            
            # Tabla comparativa
            st.markdown("#### Tabla Comparativa")
            
            comp_data = pd.DataFrame({
                'Métrica': ['Ciclos Totales', 'Valor Inicial', 'Valor Final', 'Cambio Total', 
                           'Valor Mínimo', 'Valor Máximo', 'Promedio', 'Desv. Estándar'],
                f'Motor {motor_1}': [
                    len(df_motor1),
                    f"{df_motor1[sensor].iloc[0]:.3f}",
                    f"{df_motor1[sensor].iloc[-1]:.3f}",
                    f"{df_motor1[sensor].iloc[-1] - df_motor1[sensor].iloc[0]:.3f}",
                    f"{df_motor1[sensor].min():.3f}",
                    f"{df_motor1[sensor].max():.3f}",
                    f"{df_motor1[sensor].mean():.3f}",
                    f"{df_motor1[sensor].std():.3f}",
                ],
                f'Motor {motor_2}': [
                    len(df_motor2),
                    f"{df_motor2[sensor].iloc[0]:.3f}",
                    f"{df_motor2[sensor].iloc[-1]:.3f}",
                    f"{df_motor2[sensor].iloc[-1] - df_motor2[sensor].iloc[0]:.3f}",
                    f"{df_motor2[sensor].min():.3f}",
                    f"{df_motor2[sensor].max():.3f}",
                    f"{df_motor2[sensor].mean():.3f}",
                    f"{df_motor2[sensor].std():.3f}",
                ]
            })
            
            st.dataframe(helpers.sanitize_dataframe_for_display(comp_data), width='stretch', hide_index=True)
        else:
            st.warning("Selecciona dos motores diferentes para comparar")
        
        st.divider()
        
        # 
        # RESUMEN POR SENSOR
        # 
        
        with st.expander("Ver Resumen de Todos los Sensores", expanded=False):
            st.markdown("### Variabilidad de Todos los Sensores")
            
            # Excluir configuraciones operacionales y otras columnas no-sensor
            exclude_cols = {'unit_id', 'time_cycles', 'RUL', 'rul', 'motor', 'ciclo', 
                          'config1', 'config2', 'config3',
                          'op_setting_1', 'op_setting_2', 'op_setting_3'}
            sensor_cols = [col for col in df.columns if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)]
            sensor_stats = []
            
            for s in sensor_cols:
                sensor_name = config.SENSOR_SHORT_NAMES.get(s, s)
                sensor_stats.append({
                    'Sensor': f"{sensor_name} ({s})",
                    'Promedio': df[s].mean(),
                    'Desv. Est.': df[s].std(),
                    'Mínimo': df[s].min(),
                    'Máximo': df[s].max(),
                    'Rango': df[s].max() - df[s].min(),
                    'Coef. Var. (%)': (df[s].std() / df[s].mean() * 100) if df[s].mean() != 0 else 0
                })
            
            stats_df = pd.DataFrame(sensor_stats)
            stats_df = stats_df.sort_values('Coef. Var. (%)', ascending=False)
            
            st.dataframe(helpers.sanitize_dataframe_for_display(stats_df).style.format({
                'Promedio': '{:.3f}',
                'Desv. Est.': '{:.3f}',
                'Mínimo': '{:.3f}',
                'Máximo': '{:.3f}',
                'Rango': '{:.3f}',
                'Coef. Var. (%)': '{:.2f}%'
            }).background_gradient(subset=['Coef. Var. (%)'], cmap='RdYlGn_r'), 
            width='stretch', hide_index=True)
        


