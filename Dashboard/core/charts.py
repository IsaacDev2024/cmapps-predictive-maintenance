# core/charts.py
# ------------------------------------------------------
# Módulo de visualización: funciones para crear figuras,
# ya sea con Matplotlib o Plotly, a partir de datos filtrados.
# ------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt


from core import helpers, theme

# paleta base Set2 (colores pastel)
SET2 = plt.get_cmap("Set2").colors

def get_theme_colors():
    """
    Detecta automáticamente el tema actual de Streamlit y retorna los colores apropiados.
    Retorna un diccionario con colores para texto, grid, etc.
    """
    # Detectar el tema automáticamente desde la configuración de Streamlit
    theme_base = st.session_state.get('app_theme') or theme.get_current_theme()
    
    # Retornar colores según el tema detectado
    if theme_base == "dark":
        return {
            'text': '#FFFFFF',
            'grid': '#555555',
            'spine': '#888888',
            'legend_bg': '#1E1E1EEE',
            'legend_edge': '#555555',
            'legend_text': '#FFFFFF',
            'title': '#FFFFFF',
            'edge': 'white',
            'gauge_center': '#1E1E1E'
        }
    else:  # light mode
        return {
            'text': '#000000',  # Negro puro para mejor visibilidad en modo claro
            'grid': '#CCCCCC',
            'spine': '#666666',
            'legend_bg': '#FFFFFFEE',
            'legend_edge': '#CCCCCC',
            'legend_text': '#000000',  # Negro puro
            'title': '#000000',  # Negro puro
            'edge': 'black',
            'gauge_center': 'white'
        }

def create_figure(figsize=(12, 6.5)):
    """crea una figura base transparente."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="none")
    ax.set_facecolor("none")
    return fig, ax


def style_axes(ax):
    """estilo adaptativo según el tema actual (light/dark mode)."""
    colors = get_theme_colors()
    
    # etiquetas y ticks
    ax.tick_params(colors=colors['text'], labelsize=10)
    ax.xaxis.label.set_color(colors['text'])
    ax.yaxis.label.set_color(colors['text'])

    # bordes de ejes
    for spine in ax.spines.values():
        spine.set_edgecolor(colors['spine'])
        spine.set_alpha(0.6)

    # grid
    ax.grid(True, alpha=0.35, color=colors['grid'], linestyle="--")



def plot_original(ax, x, y, label=None, xlabel=None, ylabel=None, color=None, linestyle="-", linewidth=2.2, alpha=0.9):
    """
    Dibuja una serie genérica en el eje principal.
    Parámetros:
        label: texto de la leyenda
        xlabel, ylabel: etiquetas de ejes
        color, linestyle, linewidth, alpha: estilo de línea
    """
    ax.plot(
        x, y,
        color=color or SET2[0],
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label or "Serie",
    )

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    style_axes(ax)
    return ax


def plot_stationary(ax, x_stat, stat_series, label="Estacionaria"):
    """dibuja la serie estacionaria."""
    ax2 = ax.twinx()
    plot_original(
        ax2,
        x_stat,
        stat_series,
        label=label,
        color=SET2[1],
        linestyle="--",
        linewidth=2.0,
        alpha=0.9,
    )
    ax2.grid(False)
    return ax2

def combine_legends(ax, ax2=None, loc="upper left"):
    """combina leyendas con estilo adaptativo según tema."""
    colors = get_theme_colors()
    
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    legend = ax.legend(
        all_handles,
        all_labels,
        loc=loc,
        frameon=True,
        framealpha=0.9,
        facecolor=colors['legend_bg'],
        edgecolor=colors['legend_edge'],
        fontsize=9,
        labelspacing=0.4,
    )
    for text in legend.get_texts():
        text.set_color(colors['legend_text'])
    return ax


def finalize_figure(ax, fig, title):
    """añade título y aplica ajustes finales con colores adaptativos."""
    colors = get_theme_colors()
    
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        color=colors['title'],
        pad=15,
    )
    fig.tight_layout()
    return fig


def make_evolution_figure(df_unit, sensor, stationary_map, unit_id, sensor_label=None):
    """Gráfico de evolución del sensor con eje doble y leyenda combinada."""

    fig, ax = create_figure()
    helpers.ensure_columns(df_unit, ["time_cycles", sensor])

    display_name = sensor_label or sensor

    x = df_unit["time_cycles"].reset_index(drop=True)
    y = df_unit[sensor].reset_index(drop=True).astype(float)

    # --- serie original ---
    plot_original(ax, x, y, label=f"Original — {display_name}",
                  xlabel="Ciclos", ylabel="Valor del sensor")

    ax2 = None

    # --- serie estacionaria ---
    if stationary_map is not None:
        unit_key = helpers.pick_unit_key(stationary_map, unit_id)
        if unit_key is not None:
            unit_map = stationary_map[unit_key]
            sensor_key = helpers.pick_sensor_key(unit_map, sensor)
            if sensor_key in unit_map:
                stat_series = pd.Series(unit_map[sensor_key]).reset_index(drop=True).astype(float)
                x_stat = helpers.align_x(x, len(stat_series))
                ax2 = plot_stationary(ax, x_stat, stat_series,
                                      label=f"Estacionaria — {display_name}")

    # leyenda y título
    combine_legends(ax, ax2)
    finalize_figure(ax, fig, f"Evolución del sensor '{display_name}' — Unidad {unit_id}")

    return fig


def make_behavior_plot(df, sensor, sensor_label=None):
    """Gráfico del comportamiento de un sensor por motor con tendencia polinómica (sin leyenda)."""

    fig, ax = create_figure()

    if "motor" not in df.columns or "ciclo" not in df.columns:
        raise KeyError("El DataFrame debe contener las columnas 'motor' y 'ciclo'.")

    display_name = sensor_label or sensor

    # --- líneas por motor ---
    for i, motor in enumerate(df["motor"].unique()):
        subset = df[df["motor"] == motor]
        plot_original(
            ax,
            subset["ciclo"],
            subset[sensor],
            label=None,            # sin etiqueta
            xlabel="Ciclo",
            ylabel=display_name,
            color=SET2[i % len(SET2)],
            alpha=0.35,
            linewidth=1.4,
        )

    # --- tendencia polinómica global ---
    x = df["ciclo"]
    y = df[sensor]
    if len(x) > 2:
        coef = np.polyfit(x, y, 2)
        poly = np.poly1d(coef)
        x_tend = np.linspace(x.min(), x.max(), 150)
        y_tend = poly(x_tend)
        plot_original(
            ax,
            x_tend,
            y_tend,
            label=None,            # sin etiqueta
            color=SET2[1],
            linestyle="--",
            linewidth=2.5,
            alpha=0.9,
        )

    # --- estilo general y título ---
    style_axes(ax)
    finalize_figure(ax, fig, f"Comportamiento del sensor '{display_name}' por motor")

    return fig


# 4. Utilidad auxiliar
def show_chart(fig):
    st.pyplot(fig, transparent=True, width="stretch")


def apply_adaptive_style(ax, title=None, xlabel=None, ylabel=None, show_legend=True):
    """
    Aplica estilo adaptativo completo a un eje matplotlib.
    
    Args:
        ax: eje matplotlib
        title: título del gráfico (opcional)
        xlabel: etiqueta del eje X (opcional)
        ylabel: etiqueta del eje Y (opcional)
        show_legend: si mostrar leyenda con estilo adaptativo
    """
    colors = get_theme_colors()
    
    # Aplicar estilo base
    style_axes(ax)
    
    # Título
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', 
                    color=colors['title'], pad=15)
    
    # Etiquetas de ejes
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=12, 
                     color=colors['text'])
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12, 
                     color=colors['text'])
    
    # Leyenda
    if show_legend and ax.get_legend_handles_labels()[0]:
        legend = ax.legend(frameon=True, framealpha=0.9,
                          facecolor=colors['legend_bg'],
                          edgecolor=colors['legend_edge'])
        for text in legend.get_texts():
            text.set_color(colors['legend_text'])
    
    return ax
