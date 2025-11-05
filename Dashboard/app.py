# app.py
# 
# DASHBOARD DE MANTENIMIENTO PREDICTIVO - MOTORES JET NASA C-MAPSS
# 

import streamlit as st
from core import config, theme
import base64
from pathlib import Path

# Importar vistas
from views.overview import View as OverviewView
from views.evolution import View as EvolutionView
from views.behavior import View as BehaviorView
from views.dataframe import View as DataframeView
from views.model import View as ModelView

# 
# CONFIGURACIÓN DE LA PÁGINA
# 

st.set_page_config(
    page_title=" Mantenimiento Predictivo - NASA C-MAPSS",
    page_icon="nasa-logo.svg",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com',
        'Report a bug': "https://github.com",
        'About': """
        # Dashboard de Mantenimiento Predictivo
        
        **Análisis de Supervivencia y Degradación de Motores Jet**
        
        Desarrollado por:
        - Isaac David Sánchez Sánchez
        - Germán Eduardo de Armas Castaño
        - Katlyn Gutiérrez Cardona
        - Shalom Jhoanna Arrieta Marrugo
        
        Universidad Tecnológica de Bolívar - 2025
        """
    }
)

# Aplicar CSS personalizado
st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)

# Guardar el tema activo (light/dark) para uso global
theme.get_current_theme()

# 
# SIDEBAR - NAVEGACIÓN
# 

# Función para cargar el logo de NASA
@st.cache_data
def load_nasa_logo():
    """Carga y codifica el logo de NASA en base64"""
    logo_path = Path(__file__).parent / "nasa-logo.svg"
    if logo_path.exists():
        with open(logo_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
            # Codificar SVG en base64
            b64 = base64.b64encode(svg_content.encode('utf-8')).decode()
            return f"data:image/svg+xml;base64,{b64}"
    return None

with st.sidebar:
    # Logo y título con imagen de NASA
    logo_data = load_nasa_logo()
    
    if logo_data:
        st.markdown(f"""
            <div style='text-align: center; padding: 1.5rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 border-radius: 15px; margin-bottom: 2rem;'>
                <img src="{logo_data}" style='width: 120px; margin-bottom: 0.5rem;'>
                <p style='color: white; margin: 0.5rem 0 0 0; font-size: 0.9rem; font-weight: 600;'>
                    NASA C-MAPSS
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback si no se encuentra el logo
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 border-radius: 15px; margin-bottom: 2rem;'>
                <svg width="80" height="80" viewBox="0 0 24 24" fill="white" stroke="white" stroke-width="1">
                    <path d="M12 2L2    7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                </svg>
                <p style='color: white; margin: 0.5rem 0 0 0; font-size: 0.9rem; font-weight: 600;'>
                    NASA C-MAPSS
                </p>
            </div>
        """, unsafe_allow_html=True)

# 
# REGISTRO DE VISTAS
# 

VIEWS = {
    "Dashboard Principal": OverviewView(),
    "Evolución de Sensores": EvolutionView(),
    "Comportamiento por Motor": BehaviorView(),
    "Explorador de Datos": DataframeView(),
    "Modelo LSTM - Predicciones": ModelView(),
}

# 
# SELECTOR DE VISTA
# 

# Inicializar el estado de la vista seleccionada
if 'selected_view' not in st.session_state:
    st.session_state.selected_view = "Dashboard Principal"

with st.sidebar:
    # CSS para botones elegantes del menú de navegación
    st.markdown("""
        <style>
        /* Estilo para el sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }
        
        /* Botones de navegación */
        div.stButton > button {
            width: 100% !important;
            padding: 0.75rem 1rem !important;
            margin: 0.25rem 0 !important;
            border-radius: 12px !important;
            background: rgba(255, 255, 255, 0.08) !important;
            color: #E0E0E0 !important;
            border: 2px solid transparent !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            text-align: left !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            cursor: pointer !important;
        }
        
        div.stButton > button:hover {
            background: rgba(103, 126, 234, 0.3) !important;
            border-color: rgba(103, 126, 234, 0.7) !important;
            transform: translateX(8px) !important;
            box-shadow: 0 4px 20px rgba(103, 126, 234, 0.5) !important;
        }
        
        /* Expander personalizado */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            color: #E0E0E0 !important;
            transition: all 0.3s ease;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(255, 255, 255, 0.12);
        }
        
        /* Info box en sidebar */
        .sidebar-info-box {
            padding: 1rem;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### Menú de Navegación")
    st.markdown("---")
    
    # Crear botones de navegación simples sin iconos
    for view_name in VIEWS.keys():
        is_active = st.session_state.selected_view == view_name
        
        # Aplicar estilo especial si está activo
        if is_active:
            st.markdown(f"""
                <style>
                button[key="btn_{view_name}"] {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                    border-color: #667eea !important;
                    color: white !important;
                    box-shadow: 0 6px 25px rgba(103, 126, 234, 0.7) !important;
                    transform: translateX(8px) !important;
                }}
                </style>
            """, unsafe_allow_html=True)
        
        # Botón funcional
        if st.button(view_name, key=f"btn_{view_name}", width='stretch'):
            st.session_state.selected_view = view_name
            st.rerun()
    
    st.markdown("---")
    
# Placeholder para controles de la vista (se renderizarán desde cada vista)
# Los controles aparecerán aquí porque usan st.sidebar

# 
# RENDERIZADO DE LA VISTA SELECCIONADA
# 
# RENDERIZADO DE LA VISTA SELECCIONADA
# 

view = VIEWS[st.session_state.selected_view]
view.render()
