# 🚀 Dashboard de Mantenimiento Predictivo

Este es el dashboard interactivo del Sistema de Mantenimiento Predictivo para Motores Jet NASA C-MAPSS.

## 🌐 Acceso al Dashboard

👉 **[Ver Dashboard en Vivo](https://tu-app.streamlit.app)** *(Actualizar con tu URL)*

## 📦 Instalación Local

```bash
# Instalar dependencias
pip install -r ../requirements.txt

# Ejecutar dashboard
streamlit run app.py
```

## 📊 Características

- **Vista General**: Estadísticas y resumen del dataset
- **Evolución Temporal**: Curvas de supervivencia y degradación
- **Comportamiento de Sensores**: Análisis de correlaciones
- **Datos Tabulares**: Exploración de datos
- **Predicciones LSTM**: Sistema de predicción en tiempo real

## 🔧 Configuración

El dashboard usa el archivo `.streamlit/config.toml` para la configuración de tema y servidor.

## 📁 Estructura

```
Dashboard/
├── app.py              # Punto de entrada
├── core/               # Módulos principales
├── views/              # Vistas del dashboard
├── data/               # Datos y modelos
└── .streamlit/         # Configuración de Streamlit
```

Para más información, consulta el [README principal](../README.md) del proyecto.
