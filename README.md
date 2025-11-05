<div align="center">

# 🚀 Sistema de Mantenimiento Predictivo para Motores Jet

### **Análisis Avanzado con Deep Learning sobre Dataset NASA C-MAPSS**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img src="https://img.shields.io/badge/Status-Production-success" alt="Status">
<img src="https://img.shields.io/badge/Maintained-Yes-brightgreen" alt="Maintained">

### 🌐 **[Ver Dashboard en Vivo →](https://cmapps-predictive-maintenance.streamlit.app/)**

---

### **Desarrollado por**
**Isaac David Sánchez Sánchez** • **Germán Eduardo de Armas Castaño**  
**Katlyn Gutiérrez Cardona** • **Shalom Jhoanna Arrieta Marrugo**

*Universidad Tecnológica de Bolívar - 2025*

</div>

---

## Tabla de Contenidos

- [🎯 Descripción del Proyecto](#-descripción-del-proyecto)
- [✨ Características Principales](#-características-principales)
- [🏗️ Arquitectura del Sistema](#️-arquitectura-del-sistema)
- [🧠 Modelo LSTM](#-modelo-lstm)
- [📊 Dashboard Interactivo](#-dashboard-interactivo)
- [🚀 Instalación y Configuración](#-instalación-y-configuración)
- [💡 Uso del Sistema](#-uso-del-sistema)
- [📁 Estructura del Proyecto](#-estructura-del-proyecto)
- [🔬 Análisis Exploratorio](#-análisis-exploratorio)
- [📈 Resultados y Métricas](#-resultados-y-métricas)
- [🛠️ Tecnologías Utilizadas](#️-tecnologías-utilizadas)
- [📚 Dataset](#-dataset)
- [🤝 Contribuciones](#-contribuciones)
- [📄 Licencia](#-licencia)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un **sistema completo de mantenimiento predictivo** para motores de turbofán utilizando técnicas avanzadas de **Deep Learning** y **análisis de series temporales**. El sistema es capaz de predecir el estado de salud de motores jet basándose en datos de sensores, permitiendo la detección temprana de fallos y optimizando las estrategias de mantenimiento.

### 🎓 Contexto Académico

Desarrollado como proyecto de análisis de datos avanzado utilizando el prestigioso **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)**, un dataset ampliamente reconocido en la comunidad científica para investigación en mantenimiento predictivo.

### 🎁 Valor Agregado

- **Modelo LSTM Bidireccional** de alta precisión y recall (Importante para mantenimiento predictivo)
- **Dashboard interactivo** con visualizaciones avanzadas
- **Sistema de predicción en tiempo real** con clasificación de estados
- **Análisis exploratorio completo** con insights estadísticos
- **Documentación exhaustiva** y código modular

---

## ✨ Características Principales

### 🔮 Predicción Inteligente
- **Clasificación de estados**: Normal, Fallo Inminente
- **Modelo LSTM** con secuencias temporales de 30 ciclos
- **14 sensores optimizados** seleccionados por correlación
- **Escalado automático** de datos con MinMaxScaler

### 📊 Visualizaciones Avanzadas
- **Curvas de supervivencia** con estimadores Kaplan-Meier
- **Matrices de correlación** interactivas
- **Gráficos de evolución temporal** de sensores
- **Análisis de distribuciones** y comportamiento de datos

### 🎯 Dashboard Profesional
- **Interface moderna** construida con Streamlit
- **Tema personalizado** adaptable (modo claro/oscuro)
- **Navegación intuitiva** por secciones
- **Exportación de resultados** en múltiples formatos

### 🧪 Herramientas de Análisis
- **EDA completo** en Jupyter Notebook
- **Análisis estadístico** de 21 sensores
- **Detección de patrones** de degradación
- **Validación de modelos** con métricas exhaustivas

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                     ENTRADA DE DATOS                        │
│          (Sensores de Motor - 21 variables)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              PREPROCESAMIENTO                               │
│  • Selección de 14 sensores óptimos                         │
│  • Creación de variable objetivo                            │
│  • Normalización (MinMaxScaler)                             │
│  • Creación de secuencias (30 ciclos)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              MODELO LSTM BIDIRECCIONAL                      │
│  • Capa Bidirectional LSTM (64 unidades)                    │
│  • Dropout (0.3)                                            │
│  • Capa LSTM (32 unidades)                                  │
│  • Dropout (0.3)                                            │
│  • Capa Dense (32 unidades, L2 regularization)              │
│  • Dropout (0.2)                                            │
│  • Capa Output (Sigmoid)                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              CLASIFICACIÓN DE ESTADO                        │
│  🟢 NORMAL                                                  │
│  🔴 FALLO INMINENTE                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Modelo LSTM

### Arquitectura del Modelo

El modelo implementa una **red LSTM bidireccional** optimizada para capturar patrones temporales complejos:

```python
Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(30, 14)),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

### Características Técnicas

| Parámetro | Valor |
|-----------|-------|
| **Tipo de modelo** | LSTM Bidireccional |
| **Secuencia temporal** | 30 ciclos |
| **Características (sensores)** | 14 variables |
| **Función de pérdida** | Binary Crossentropy |
| **Optimizador** | Adam |
| **Regularización** | L2 (0.01) + Dropout (0.2-0.3) |
| **Métricas** | Accuracy, Precision, Recall, AUC |

### Sensores Utilizados

El modelo utiliza los **14 sensores más correlacionados** con el estado del motor:

```
T24, T30, T50, P30, Nf, Nc, Ps30, phi, NRf, NRc, BPR, htBleed, W31, W32
```

*Seleccionados mediante análisis de correlación (threshold ≥ 0.2)*

---

## 📊 Dashboard Interactivo

### 🏠 Vistas Disponibles

#### 1. **📈 Overview (Visión General)**
- Resumen estadístico del dataset
- Métricas principales de los motores
- Distribución de ciclos de vida
- Matriz de correlación general

#### 2. **🔄 Evolution (Evolución Temporal)**
- Curvas de supervivencia Kaplan-Meier
- Análisis gráfico y estadístico de degradación temporal por motor y por sensor
- Patrones de fallo

#### 3. **🎯 Behavior (Comportamiento de Sensores)**
- Análisis general por sensores
- Análisis de vida útil restante de los motores
- Patrones de comportamiento
- Comparación entre motores

#### 4. **📋 DataFrame (Datos Crudos)**
- Exploración de datos tabulares
- Filtros avanzados
- Búsqueda por motor y ciclo
- Exportación de datos

#### 5. **🤖 Model (Predicciones)**
- **Sistema de predicción interactivo**
- Carga de datos de motores
- Clasificación de estado en tiempo real
- Visualización de resultados
- Interpretación de predicciones

### 🎨 Características del Dashboard

- **Tema personalizado** con gradientes modernos
- **Navegación fluida** entre secciones
- **Gráficos interactivos** con Plotly
- **Responsive design** adaptable
- **Código modular** y mantenible

---

## 🚀 Instalación y Configuración

### Requisitos Previos

- **Python 3.9+**
- **pip** (gestor de paquetes)
- **Git** (opcional, para clonar el repositorio)

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/cmapps.git
cd cmapps
```

### Paso 2: Crear Entorno Virtual (Recomendado)

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En macOS/Linux:
source venv/bin/activate

# En Windows:
venv\Scripts\activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

#### Librerías Principales

```
tensorflow>=2.10.0
streamlit>=1.25.0
pandas>=1.5.0
numpy>=1.23.0
plotly>=5.14.0
scikit-learn>=1.2.0
lifelines>=0.27.0
seaborn>=0.12.0
matplotlib>=3.7.0
joblib>=1.2.0
```

### Paso 4: Verificar Instalación

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
streamlit --version
```

---

## 💡 Uso del Sistema

### 🎯 Ejecutar el Dashboard

```bash
cd Dashboard
streamlit run app.py
```

El dashboard se abrirá automáticamente en tu navegador en `http://localhost:8501`

### 📓 Ejecutar el Notebook de Análisis

```bash
jupyter notebook "Notebook - EDA y Modelado LSTM.ipynb"
```

O abrirlo directamente en **VS Code** con la extensión de Jupyter.

### 🔮 Hacer Predicciones

#### Opción 1: Usar el Dashboard (Recomendado)

1. Navega a la sección **"🤖 Model"**
2. Carga un archivo CSV con datos del motor
3. El sistema automáticamente:
   - Valida el formato
   - Selecciona los sensores correctos
   - Normaliza los datos
   - Genera predicciones
   - Clasifica el estado del motor

#### Opción 2: Usar Python Directamente

```python
from tensorflow import keras
from joblib import load
import numpy as np

# Cargar modelo y escalador
modelo = keras.models.load_model('modelos_lstm/modelo_lstm_completo.keras')
scaler = load('modelos_lstm/scaler_lstm.bin')

# Preparar datos (ejemplo)
# X debe tener shape: (n_secuencias, 30, 14)
X_nuevos = scaler.transform(datos_sensores)
X_secuencias = crear_secuencias(X_nuevos, seq_length=30)

# Hacer predicción
predicciones = modelo.predict(X_secuencias)

# Clasificar estado
def clasificar_estado(prob):
    if prob > 0.75:
        return "🟢 NORMAL"
    elif prob > 0.35:
        return "🟡 ALERTA"
    else:
        return "🔴 FALLO INMINENTE"

estado = clasificar_estado(predicciones[-1][0])
print(f"Estado del motor: {estado}")
```

### 📁 Formato de Datos de Entrada
**Requisitos:**

Para hacer predicciones, tu archivo CSV debe incluir:

- Mínimo **30 ciclos** (filas)
- **26 columnas:** 1. unit_id | 2. time_cycles | 3-5. op_setting (3 columnas) | 6-26. sensores (21 columnas: T2, T24, T30, T50, P2, P15, P30, Nf, Nc, epr, Ps30, phi, NRf, NRc, BPR, farB, htBleed, Nf_dmd, PCNfR_dmd, W31, W32)

---

## 📁 Estructura del Proyecto

```
cmapps/
│
├── 📓 Notebook - EDA y Modelado LSTM.ipynb    # Análisis completo y entrenamiento
│
├── 📊 Dashboard/                               # Aplicación web interactiva
│   ├── app.py                                  # Punto de entrada principal
│   │
│   ├── core/                                   # Módulos base
│   │   ├── charts.py                           # Funciones de visualización
│   │   ├── config.py                           # Configuración global
│   │   ├── data.py                             # Carga y procesamiento de datos
│   │   ├── helpers.py                          # Funciones auxiliares
│   │   ├── theme.py                            # Gestión de temas
│   │   └── ui.py                               # Componentes UI
│   │
│   ├── views/                                  # Vistas del dashboard
│   │   ├── overview.py                         # Vista general
│   │   ├── evolution.py                        # Evolución temporal
│   │   ├── behavior.py                         # Comportamiento de sensores
│   │   ├── dataframe.py                        # Datos tabulares
│   │   └── model.py                            # Predicciones LSTM
│   │
│   └── data/                                   # Datos y modelos
│       ├── csv_train.csv                       # Dataset de entrenamiento
│       ├── train_FD001.txt                     # Datos originales NASA
│       └── modelo/
│           └── modelo_lstm_completo.keras      # Modelo entrenado
│
├── 🗂️ dataset/                                 # Datasets originales NASA
│   ├── train_FD001.txt                         # Datos de entrenamiento
│   ├── test_FD001.txt                          # Datos de prueba
│   └── RUL_FD001.txt                           # Remaining Useful Life
│
├── 🧠 modelos_lstm/                            # Modelos guardados
│   ├── modelo_lstm_completo.keras              # Modelo completo
│   ├── modelo_lstm_pesos.weights.h5            # Pesos del modelo
│   ├── scaler_lstm.bin                         # Escalador entrenado
│   └── README_CARGA_MODELO.txt                 # Instrucciones de carga
│
├── 🖼️ images/                                  # Imágenes y gráficos
│
├── 📄 README.md                                # Este archivo
├── 📄 requirements.txt                         # Dependencias del proyecto
└── 📄 LICENSE                                  # Licencia del proyecto
```

---

## 🔬 Análisis Exploratorio

El notebook `Notebook - EDA y Modelado LSTM.ipynb` contiene un análisis exhaustivo que incluye:

### 📊 Análisis Estadístico
- Distribución de ciclos de vida por motor
- Estadísticas descriptivas de 21 sensores
- Identificación de valores atípicos
- Análisis de variabilidad

### 🔗 Análisis de Correlaciones
- Matriz de correlación completa
- Selección de características óptimas
- Análisis de multicolinealidad
- Feature importance

### 📈 Análisis de Supervivencia
- Curvas de Kaplan-Meier
- Estimación de funciones de riesgo
- Análisis de censura
- Comparación entre grupos

### 🎯 Análisis Temporal
- Patrones de degradación
- Tendencias de sensores
- Detección de puntos de cambio
- Análisis de estacionalidad

### 🧪 Validación de Modelo
- Separación train/test estratificada
- Validación cruzada temporal
- Métricas de rendimiento
- Análisis de errores

---

## 🛠️ Tecnologías Utilizadas

### 🐍 Lenguaje Principal
- **Python 3.9+**

### 🧠 Machine Learning & Deep Learning
- **TensorFlow / Keras** - Modelo LSTM
- **scikit-learn** - Preprocesamiento y métricas
- **lifelines** - Análisis de supervivencia

### 📊 Análisis de Datos
- **Pandas** - Manipulación de datos
- **NumPy** - Computación numérica
- **SciPy** - Análisis estadístico

### 📈 Visualización
- **Plotly** - Gráficos interactivos
- **Matplotlib** - Visualizaciones estáticas
- **Seaborn** - Visualizaciones estadísticas

### 🌐 Web Framework
- **Streamlit** - Dashboard interactivo
- **HTML/CSS** - Personalización de UI

### 🔧 Utilidades
- **Joblib** - Serialización de modelos
- **Jupyter** - Notebooks interactivos

---

## 📚 Dataset

### NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) - FD001

El dataset proviene del **NASA Ames Prognostics Data Repository** y simula el comportamiento de motores de turbofán bajo la condición operacional (Sea Level) y la condición de fallo (HPC Degradation). 

#### Características del Dataset

| Característica | Descripción |
|----------------|-------------|
| **Tipo** | Series temporales multivariadas |
| **Motores** | 100 unidades (FD001) |
| **Sensores** | 21 variables de sensores |
| **Configuraciones** | 3 parámetros operacionales |
| **Ciclos promedio** | ~200 por motor |
| **Condición de fallo** | Degradación progresiva hasta fallo |

#### Variables del Dataset

- **motor**: ID único del motor (1-100)
- **ciclo**: Ciclo operacional (tiempo discreto)
- **config1, config2, config3**: Configuraciones operacionales
- **sensor1 - sensor21**: Lecturas de 21 sensores diferentes
  - Sensores de temperatura
  - Sensores de presión
  - Sensores de velocidad
  - Sensores de flujo
  - Otros parámetros operacionales

#### Referencias

```
A. Saxena, K. Goebel, D. Simon, and N. Eklund (2008). 
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", 
International Conference on Prognostics and Health Management, Denver, CO.
```

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar este proyecto:

1. **Fork** el repositorio
2. Crea una **rama** para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. Abre un **Pull Request**

### 🐛 Reportar Bugs

Si encuentras un bug, por favor abre un **issue** describiendo:
- Descripción del problema
- Pasos para reproducir
- Comportamiento esperado vs actual
- Screenshots (si aplica)
- Entorno (OS, versión de Python, etc.)

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

```
MIT License

Copyright (c) 2025 Universidad Tecnológica de Bolívar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 📞 Contacto

### Equipo de Desarrollo

- **Isaac David Sánchez Sánchez** - [GitHub](https://github.com/IsaacDev2024)
- **Germán Eduardo de Armas Castaño**
- **Katlyn Gutiérrez Cardona**
- **Shalom Jhoanna Arrieta Marrugo**

### Institución

**Universidad Tecnológica de Bolívar**  
Cartagena de Indias, Colombia  
[www.utb.edu.co](https://www.utb.edu.co)

---

<div align="center">

### ⭐ Si este proyecto te fue útil, considera darle una estrella ⭐

**Hecho con ❤️ en Colombia 🇨🇴**

---

*"La predicción no es sólo sobre el futuro, es sobre tomar decisiones informadas hoy."*

</div>
