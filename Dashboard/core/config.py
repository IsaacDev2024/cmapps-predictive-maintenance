# core/config.py
# ===============================================
# Configuraciones centralizadas del Dashboard
# ===============================================

# 
# PALETA DE COLORES
# 

COLORS = {
    # Colores principales
    'primary': '#2E86AB',      # Azul principal
    'secondary': '#A23B72',    # Rosa/Púrpura
    'accent': '#F18F01',       # Naranja
    'success': '#06A77D',      # Verde
    'warning': '#F77F00',      # Amarillo/Naranja
    'danger': '#D62828',       # Rojo
    'info': '#4ECDC4',         # Turquesa
    
    # Tonos neutros
    'dark': '#2B2D42',
    'gray_dark': '#555555',
    'gray': '#8D99AE',
    'gray_light': '#EDF2F4',
    'white': '#FFFFFF',
    
    # Gradientes para gráficos
    'gradient_blue': ['#0A2463', '#247BA0', '#3DBAC4', '#9ED8DB'],
    'gradient_warm': ['#D62828', '#F77F00', '#FCBF49', '#06A77D'],
    'gradient_purple': ['#2E1760', '#6A2C70', '#9F5F80', '#F08A4B'],
    
    # Estados
    'normal': '#06A77D',       # Motor normal
    'warning_state': '#F77F00', # Alerta
    'failure': '#D62828',      # Fallo inminente
}

# 
# CONFIGURACIÓN DE GRÁFICOS
# 

CHART_CONFIG = {
    'figure_size': (14, 7),
    'figure_size_large': (16, 8),
    'figure_size_small': (10, 5),
    'dpi': 100,
    'title_size': 16,
    'label_size': 12,
    'tick_size': 10,
    'legend_size': 10,
    'grid_alpha': 0.3,
    'line_width': 2.5,
    'line_alpha': 0.9,
}

# 
# PARÁMETROS DEL MODELO
# 

MODEL_CONFIG = {
    'failure_threshold': 20,    # Umbral de ciclos para fallo inminente
    'sequence_length': 30,      # Longitud de secuencias LSTM
    'correlation_threshold': 0.2, # Umbral mínimo de correlación
}

# 
# RUTAS DE ARCHIVOS
# 

PATHS = {
    'train_data': 'data/train_FD001.txt',
    'csv_train': 'data/csv_train.csv',
    'stationary_pickle': 'data/all_units_stationary.pkl',
    'model_lstm': 'data/modelo/modelo_lstm_completo.keras',
    'scaler_lstm': 'data/modelo/scaler_lstm.bin',
    'history_lstm': 'data/modelo/historial_entrenamiento_lstm.pkl',
}

# 
# NOMBRES DE COLUMNAS
# 

COLUMN_NAMES = [
    'motor', 'ciclo', 'config1', 'config2', 'config3', 'sensor1',
    'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6', 'sensor7',
    'sensor8', 'sensor9', 'sensor10', 'sensor11', 'sensor12', 'sensor13',
    'sensor14', 'sensor15', 'sensor16', 'sensor17', 'sensor18', 'sensor19',
    'sensor20', 'sensor21'
]

# Columnas a excluir de análisis
EXCLUDE_COLUMNS = {'motor', 'ciclo', 'config1', 'config2', 'config3', 'RUL', 'rul', 'estado'}

# Nombres cortos de sensores para visualización compacta
SENSOR_SHORT_NAMES = {
    'sensor1': 'T2',
    'sensor2': 'T24',
    'sensor3': 'T30',
    'sensor4': 'T50',
    'sensor5': 'P2',
    'sensor6': 'P15',
    'sensor7': 'P30',
    'sensor8': 'Nf',
    'sensor9': 'Nc',
    'sensor10': 'epr',
    'sensor11': 'Ps30',
    'sensor12': 'phi',
    'sensor13': 'NRf',
    'sensor14': 'NRc',
    'sensor15': 'BPR',
    'sensor16': 'farB',
    'sensor17': 'htBleed',
    'sensor18': 'Nf_dmd',
    'sensor19': 'PCNfR_dmd',
    'sensor20': 'W31',
    'sensor21': 'W32',
}

# 
# MENSAJES Y TEXTOS
# 

MESSAGES = {
    'loading': 'Cargando datos...',
    'processing': 'Procesando...',
    'success': 'Operación completada exitosamente',
    'error': 'Error en la operación',
    'warning': 'Advertencia',
    'info': 'Información',
}

# 
# ESTILOS PERSONALIZADOS
# 

CUSTOM_CSS = """
<style>
    /* 
       MODO CLARO - Configuración por defecto
        */
    
    /* Fondo adaptativo */
    .main {
        background-color: #F8F9FA;
    }
    
    /* Títulos con alto contraste en modo claro */
    h1 {
        color: #2E86AB !important;
        font-weight: 700;
        padding-bottom: 1rem;
        border-bottom: 3px solid #2E86AB;
        margin-bottom: 2rem;
    }
    
    h2 {
        color: #1a1a1a !important;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: #1a1a1a !important;
        font-weight: 500;
    }
    
    h4 {
        color: #2B2D42 !important;
    }
    
    /* Texto general modo claro - Negro para máxima legibilidad */
    .stMarkdown p {
        color: #1a1a1a;
    }
    
    .stMarkdown span {
        color: inherit;
    }
    
    .stMarkdown div {
        color: inherit;
    }
    
    /* Labels y textos de UI */
    label {
        color: #1a1a1a !important;
    }
    
    /* Texto en métricas */
    [data-testid="stMetricLabel"] {
        color: #1a1a1a !important;
        font-weight: 600;
    }
    
    /* 
       MODO OSCURO - Overrides específicos
        */
    
    @media (prefers-color-scheme: dark) {
        .main {
            background-color: #0E1117;
        }
        
        h1 {
            color: #4ECDC4 !important;
            border-bottom-color: #4ECDC4;
        }
        
        h2 {
            color: #FFFFFF !important;
        }
        
        h3 {
            color: #FFFFFF !important;
        }
        
        h4 {
            color: #E0E0E0 !important;
        }
        
        .stMarkdown p {
            color: #E0E0E0;
        }
        
        .stMarkdown span {
            color: inherit;
        }
        
        .stMarkdown div {
            color: inherit;
        }
        
        /* KPI Cards en modo oscuro */
        .kpi-card {
            background: #1E1E1E;
            border-left-color: #4ECDC4;
        }
        
        .kpi-value {
            color: #FFFFFF !important;
        }
        
        .kpi-subtitle {
            color: #B0B0B0 !important;
        }
    }
    
    /* Métricas destacadas - Modo Claro */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #2E86AB !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
        color: #1a1a1a !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
        font-weight: 600 !important;
        color: #555555 !important;
    }
    
    /* Métricas en Dark Mode */
    @media (prefers-color-scheme: dark) {
        [data-testid="stMetricValue"] {
            color: #4ECDC4 !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #FFFFFF !important;
        }
        
        [data-testid="stMetricDelta"] {
            color: #B0B0B0 !important;
        }
    }
    
    /* Cajas de sensores - modo claro */
    .sensor-box {
        padding: 0.5rem;
        margin: 0.25rem 0;
        background: #EDF2F4;
        border-radius: 5px;
        color: #1a1a1a !important;
    }
    
    .sensor-box b {
        color: #1a1a1a !important;
    }
    
    .sensor-box-success {
        border-left: 4px solid #06A77D;
    }
    
    .sensor-box-warning {
        border-left: 4px solid #F77F00;
    }
    
    /* Caja de insights/información */
    .info-box-insight {
        padding: 0.75rem;
        background: #EDF2F4;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        color: #1a1a1a !important;
    }
    
    .info-box-insight b {
        color: #1a1a1a !important;
    }
    
    /* Tarjetas de información */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white !important;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .info-card h3, .info-card p {
        color: white !important;
    }
    
    .success-card {
        background: linear-gradient(135deg, #06A77D 0%, #04E762 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white !important;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .success-card h3, .success-card p, .success-card h4 {
        color: white !important;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #F77F00 0%, #FCBF49 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white !important;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .warning-card h3, .warning-card p {
        color: white !important;
    }
    
    .danger-card {
        background: linear-gradient(135deg, #D62828 0%, #F77F00 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white !important;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Botones */
    .stButton>button {
        background: linear-gradient(135deg, #2E86AB 0%, #4ECDC4 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2B2D42 0%, #2E86AB 100%) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Selectbox en sidebar */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: white !important;
        color: #2B2D42 !important;
        font-weight: 600 !important;
    }
    
    /* Destacar el selector de sensores */
    [data-testid="stSidebar"] label {
        font-size: 0.95rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
    }
    
    /* Primer selectbox (Unidad) con menos margen superior */
    [data-testid="stSidebar"] .stSelectbox:first-of-type {
        margin-top: 0 !important;
    }
    
    /* Reducir espacio entre selectboxes */
    [data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 0.75rem !important;
    }
    
    /* Hacer el selectbox de sensores más prominente */
    [data-testid="stSidebar"] .stSelectbox:nth-of-type(2) label {
        font-size: 1rem !important;
        font-weight: 700 !important;
        color: #FFD700 !important;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
    }
    
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #2B2D42 !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="select"] input {
        color: #2B2D42 !important;
    }
    
    [data-testid="stSidebar"] [role="button"] {
        color: #2B2D42 !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: white !important;
        color: #2B2D42 !important;
    }
    
    /* Dropdown options */
    [data-baseweb="popover"] {
        background-color: white !important;
    }
    
    [data-baseweb="popover"] li {
        color: #2B2D42 !important;
        font-weight: 500 !important;
    }
    
    [data-baseweb="popover"] li:hover {
        background-color: rgba(46, 134, 171, 0.1) !important;
        color: #2E86AB !important;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    
    .kpi-title {
        font-size: 0.9rem;
        color: #8D99AE !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .kpi-value {
        font-size: 2.5rem;
        color: #2B2D42 !important;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .kpi-subtitle {
        font-size: 0.85rem;
        color: #555555 !important;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
        color: white !important;
    }
    
    .badge-success {
        background-color: #06A77D !important;
    }
    
    .badge-warning {
        background-color: #F77F00 !important;
    }
    
    .badge-danger {
        background-color: #D62828 !important;
    }
    
    .badge-info {
        background-color: #4ECDC4 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #8D99AE !important;
        font-size: 0.9rem;
        border-top: 1px solid rgba(141, 153, 174, 0.3);
        margin-top: 3rem;
    }
    
    .footer p {
        color: #8D99AE !important;
    }
    
    /* 
        DARK MODE OVERRIDES
        */
    
    @media (prefers-color-scheme: dark) {
        /* Headers */
        h1 {
            color: #4ECDC4 !important;
        }
        
        h2 {
            color: #FFFFFF !important;
        }
        
        h3 {
            color: #FFFFFF !important;
        }
        
        h4 {
            color: #E0E0E0 !important;
        }
        
        /* Text elements */
        p, span, div, label {
            color: #E0E0E0 !important;
        }
        
        /* Expander headers - CRÍTICO para que se vean los títulos de secciones */
        [data-testid="stExpander"] summary {
            color: #FFFFFF !important;
            background-color: rgba(78, 205, 196, 0.15) !important;
        }
        
        [data-testid="stExpander"] summary:hover {
            background-color: rgba(78, 205, 196, 0.25) !important;
        }
        
        /* Markdown headers dentro de expanders */
        [data-testid="stExpander"] h1,
        [data-testid="stExpander"] h2,
        [data-testid="stExpander"] h3,
        [data-testid="stExpander"] h4 {
            color: #FFFFFF !important;
        }
        
        /* Columnas con contenido - las cajas blancas */
        [data-testid="column"] {
            background-color: rgba(43, 45, 66, 0.3) !important;
        }
        
        [data-testid="column"] p,
        [data-testid="column"] span,
        [data-testid="column"] div {
            color: #E0E0E0 !important;
        }
        
        /* Cajas de sensores en dark mode */
        .sensor-box {
            background: rgba(78, 205, 196, 0.1) !important;
            color: #E0E0E0 !important;
        }
        
        .sensor-box b {
            color: #FFFFFF !important;
        }
        
        .sensor-box-success {
            border-left-color: #06A77D;
            background: rgba(6, 167, 125, 0.15) !important;
        }
        
        .sensor-box-warning {
            border-left-color: #F77F00;
            background: rgba(247, 127, 0, 0.15) !important;
        }
        
        /* Caja de insights en dark mode */
        .info-box-insight {
            background: rgba(102, 126, 234, 0.15) !important;
            border-left-color: #667eea;
            color: #E0E0E0 !important;
        }
        
        .info-box-insight b {
            color: #FFFFFF !important;
        }
        
        .info-box-insight div {
            color: #E0E0E0 !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #4ECDC4 !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #B0B0B0 !important;
        }
        
        /* KPI Cards - mantener visibilidad en dark mode */
        .kpi-card {
            background: rgba(78, 205, 196, 0.1);
            border-left-color: #4ECDC4;
        }
        
        .kpi-title {
            color: #B0B0B0 !important;
        }
        
        .kpi-value {
            color: #4ECDC4 !important;
        }
        
        .kpi-subtitle {
            color: #B0B0B0 !important;
        }
        
        /* Info cards en dark mode */
        .info-card, .success-card, .warning-card, .danger-card {
            opacity: 0.95;
        }
        
        /* Sidebar - mantener el mismo estilo */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1A1C2E 0%, #2E86AB 100%) !important;
        }
        
        /* Footer */
        .footer {
            color: #B0B0B0 !important;
            border-top-color: rgba(176, 176, 176, 0.3);
        }
        
        .footer p {
            color: #B0B0B0 !important;
        }
    }
</style>
"""

# 
# SVG ICONOS
# 

ICONS = {
    'home': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path></svg>',
    'chart': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>',
    'evolution': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
    'behavior': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M12 1v6m0 6v6"></path></svg>',
    'model': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M12 6v6l4 2"></path></svg>',
    'data': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>',
    'settings': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M12 1v6m0 6v6m5.196-13.196l-4.242 4.242m-2.828 2.828l-4.242 4.242"></path></svg>',
    'success': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
    'error': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>',
    'warning': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
    'info': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>',
    'motor': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polygon points="10 8 16 12 10 16 10 8"></polygon></svg>',
    'sensor': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 7l-5-5-5 5M7 17l5 5 5-5"></path></svg>',
    'alert': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
    'check': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>',
}

