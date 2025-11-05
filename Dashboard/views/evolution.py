# views/evolution.py
# 
# VISTA DE EVOLUCIÓN - ANÁLISIS TEMPORAL DE SENSORES
# 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from core import ui, helpers, charts, data, config


class View:
    id = "evolution"
    label = "Evolución de Sensores"
    layout = "sidebar"

    @st.cache_data(show_spinner=False)
    def load_main_data(_self):
        """Carga y normaliza el dataset principal."""
        df = data.load_data(config.PATHS['csv_train'])
        df.columns = [c.strip() for c in df.columns] 
        return df

    @st.cache_resource(show_spinner=False)
    def load_stationary_data(_self):
        """Carga el dataset estacionario (pickle)."""
        return data.load_stationary_pickle(config.PATHS['stationary_pickle'])

    def render(self):
        """Ejecuta toda la vista (sin requerir parámetros externos)."""
        
        # Aplicar CSS
        st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)
        
        # 
        # HEADER
        # 
        
        st.title("Evolución Temporal de Sensores - FD001")
        st.markdown("""
            <div style='font-size: 1rem; color: #555555; margin-bottom: 2rem;'>
                Analiza el comportamiento de sensores individuales a lo largo del ciclo  de vida de cada motor.
                Compara la señal original con su versión estacionaria y visualiza tendencias de degradación.
                Puede seleccionar los diferentes motores y sensores en la parte inferior del Menú de Navegación.    
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()

        # 
        # CARGAR DATOS
        # 
        
        df = self.load_main_data()
        stationary_map = self.load_stationary_data()

        helpers.ensure_columns(df, ["unit_id", "time_cycles"])
        helpers.stop_if_empty(df, "El dataset está vacío o no se pudo cargar.")

        # 
        # CONTROLES INTERACTIVOS
        # 
        
        params = ui.controls_evolution(df, stationary_map, location=self.layout)
        unit_id = params.get("unit_id")
        sensor = params.get("sensor")
        sensor_label = params.get("sensor_label", sensor)
        show_stationary = params.get("show_stationary", False)

        # 
        # FILTRAR DATOS
        # 
        
        df_unit = helpers.get_unit_df(df, unit_id)
        df_unit = df_unit.sort_values("time_cycles")
        helpers.ensure_columns(df_unit, ["time_cycles", sensor])

        # 
        # KPIs Y MÉTRICAS
        # 
        
        st.markdown("### Información del Motor Seleccionado")
        
        ciclo_max = int(df_unit["time_cycles"].max())
        total_registros = len(df_unit)
        valor_inicial = df_unit[sensor].iloc[0]
        valor_final = df_unit[sensor].iloc[-1]
        cambio_total = valor_final - valor_inicial
        cambio_porcentual = (cambio_total / valor_inicial * 100) if valor_inicial != 0 else 0
        
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        
        with kpi1:
            st.metric(
                label="Motor ID",
                value=f"#{unit_id}",
                delta=None
            )
        
        with kpi2:
            st.metric(
                label=" Ciclos Totales",
                value=f"{ciclo_max}",
                delta="hasta el fallo"
            )
        
        with kpi3:
            st.metric(
                label="Sensor",
                value=sensor_label,
                delta=f"{total_registros} lecturas"
            )
        
        with kpi4:
            st.metric(
                label="Valor Inicial → Final",
                value=f"{valor_final:.2f}",
                delta=f"{cambio_total:+.2f} ({cambio_porcentual:+.1f}%)"
            )
        
        with kpi5:
            tendencia = "Ascendente" if cambio_total > 0 else "Descendente" if cambio_total < 0 else "Estable"
            st.metric(
                label="Tendencia",
                value=tendencia,
                delta=None
            )
        
        st.divider()
        
        # 
        # GRÁFICO PRINCIPAL DE EVOLUCIÓN
        # 
        
        st.markdown(f"### Evolución del Sensor **{sensor_label}** a lo Largo del Tiempo")
        
        # Verificar si el sensor es constante
        sensor_data = df_unit[sensor]
        es_constante = (sensor_data.std() == 0) or (sensor_data.nunique() == 1)
        
        if es_constante:
            st.warning(f"**Advertencia:** El sensor `{sensor_label}` tiene valores constantes (sin variabilidad) para este motor. "
                      f"Esto indica que este sensor no aporta información útil para el análisis de degradación.")
        
        fig = charts.make_evolution_figure(
            df_unit=df_unit,
            sensor=sensor,
            stationary_map=stationary_map if show_stationary else None,
            unit_id=unit_id,
            sensor_label=sensor_label,
        )

        charts.show_chart(fig)
        ui.download_button(fig, filename=f"evolucion_{sensor}_motor_{unit_id}.png")
        
        st.divider()
        
        # 
        # ANÁLISIS ESTADÍSTICO DETALLADO
        # 
        
        st.markdown("### Análisis Estadístico Detallado")
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### Estadísticas Descriptivas")
            
            sensor_data = df_unit[sensor]
            
            stats_df = pd.DataFrame({
                'Métrica': [
                    'Mínimo',
                    'Máximo',
                    'Promedio',
                    'Mediana',
                    'Desv. Estándar',
                    'Varianza',
                    'Rango',
                    'Coef. Variación (%)',
                ],
                'Valor': [
                    f"{sensor_data.min():.4f}",
                    f"{sensor_data.max():.4f}",
                    f"{sensor_data.mean():.4f}",
                    f"{sensor_data.median():.4f}",
                    f"{sensor_data.std():.4f}",
                    f"{sensor_data.var():.4f}",
                    f"{sensor_data.max() - sensor_data.min():.4f}",
                    f"{(sensor_data.std() / sensor_data.mean() * 100):.2f}%" if sensor_data.mean() != 0 else "N/A",
                ]
            })
            
            st.dataframe(helpers.sanitize_dataframe_for_display(stats_df), width='stretch', hide_index=True)
        
        with col_right:
            st.markdown("#### Distribución de Valores")
            
            fig_dist, ax = plt.subplots(figsize=(10, 5), facecolor='none')
            ax.set_facecolor('none')
            
            # Obtener colores adaptativos
            from core.charts import get_theme_colors
            theme_colors = get_theme_colors()
            
            # Histograma con KDE
            ax.hist(sensor_data, bins=30, alpha=0.6, color=config.COLORS['primary'], 
                   edgecolor=theme_colors['edge'], density=True, label='Histograma')
            
            # KDE (Kernel Density Estimation)
            sensor_data.plot(kind='density', ax=ax, color=config.COLORS['danger'], 
                            linewidth=2.5, label='Densidad (KDE)')
            
            # Líneas verticales para media y mediana
            ax.axvline(sensor_data.mean(), color=config.COLORS['success'], 
                      linestyle='--', linewidth=2, label=f'Media: {sensor_data.mean():.2f}')
            ax.axvline(sensor_data.median(), color=config.COLORS['warning'], 
                      linestyle='--', linewidth=2, label=f'Mediana: {sensor_data.median():.2f}')
            
            # Estilo adaptativo
            ax.set_xlabel('Valor del Sensor', fontweight='bold', color=theme_colors['text'])
            ax.set_ylabel('Densidad', fontweight='bold', color=theme_colors['text'])
            ax.set_title(f'Distribución de {sensor_label} - Motor #{unit_id}', fontweight='bold', pad=15, color=theme_colors['title'])
            ax.tick_params(colors=theme_colors['text'], labelsize=10)
            ax.grid(alpha=0.3, color=theme_colors['grid'])
            
            # Leyenda con fondo adaptativo
            legend = ax.legend(frameon=True, framealpha=0.9, 
                             facecolor=theme_colors['legend_bg'], edgecolor=theme_colors['legend_edge'])
            for text in legend.get_texts():
                text.set_color(theme_colors['legend_text'])
            
            plt.tight_layout()
            st.pyplot(fig_dist, transparent=True)
        
        st.divider()
        
        # 
        # ANÁLISIS DE TENDENCIA Y REGRESIÓN
        # 
        
        st.markdown("### Análisis de Tendencia Polinómica")
        
        # Calcular regresión polinomial
        x = df_unit["time_cycles"].values
        y = df_unit[sensor].values
        
        # Verificar si los datos son constantes
        y_std = np.std(y)
        is_constant = y_std < 1e-10
        
        if is_constant:
            # Datos constantes: no hay tendencia
            r_squared = 0.0
            correlation = 0.0
            p_value = 1.0
            y_fit = np.full_like(y, np.mean(y))
        else:
            # Ajuste polinomial de grado 2
            coef = np.polyfit(x, y, 2)
            poly_func = np.poly1d(coef)
            y_fit = poly_func(x)
            
            # Calcular R² con protección contra división por cero
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
            
            # Calcular correlación de Pearson
            try:
                correlation, p_value = stats.pearsonr(x, y)
            except:
                correlation, p_value = 0.0, 1.0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Coeficiente R²</div>
                    <div class="kpi-value">{r_squared:.4f}</div>
                    <div class="kpi-subtitle">Bondad de ajuste</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Correlación de Pearson</div>
                    <div class="kpi-value">{correlation:.4f}</div>
                    <div class="kpi-subtitle">Linealidad</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            significancia = "Significativa" if p_value < 0.05 else "No significativa"
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">P-value</div>
                    <div class="kpi-value">{p_value:.6f}</div>
                    <div class="kpi-subtitle">{significancia}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Interpretación
        st.markdown("#### Interpretación")
        
        if abs(correlation) > 0.7:
            corr_strength = "fuerte"
            corr_color = config.COLORS['success']
        elif abs(correlation) > 0.4:
            corr_strength = "moderada"
            corr_color = config.COLORS['warning']
        else:
            corr_strength = "débil"
            corr_color = config.COLORS['danger']
        
        direction = "positiva" if correlation > 0 else "negativa"
        
        st.markdown(f"""
            <div style='padding: 1rem; background: {corr_color}22; border-left: 4px solid {corr_color}; border-radius: 5px;'>
                <p style='margin: 0; color: #2B2D42;'>
                    <b>Correlación {corr_strength} {direction}</b> entre el tiempo y el sensor {sensor}.<br>
                    El modelo polinómico explica <b>{r_squared*100:.1f}%</b> de la variabilidad en los datos.
                    {' La correlación es estadísticamente significativa.' if p_value < 0.05 else ' La correlación NO es estadísticamente significativa.'}
                </p>
            </div>
        """, unsafe_allow_html=True)