
╔═══════════════════════════════════════════════════════════════════════╗
║        INSTRUCCIONES PARA CARGAR EL MODELO LSTM GUARDADO            ║
╚═══════════════════════════════════════════════════════════════════════╝

📅 Fecha de entrenamiento: 20251103_005810
🎯 Longitud de secuencia: 30 ciclos
📊 Características: 14

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 OPCIÓN 1: Cargar modelo completo (RECOMENDADO)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from tensorflow import keras
from joblib import load

# Cargar modelo
modelo = keras.models.load_model('modelos_lstm/modelo_lstm_completo.keras')

# Cargar escalador
scaler = load('modelos_lstm/scaler_lstm.bin')

# Hacer predicciones en nuevos datos
# X_nuevos debe tener shape: (n_secuencias, 30, 14)
# predicciones = modelo.predict(X_nuevos)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 OPCIÓN 2: Cargar solo pesos (requiere reconstruir arquitectura)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Reconstruir arquitectura
modelo = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), 
                  input_shape=(30, 14)),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Cargar pesos
modelo.load_weights('modelos_lstm/modelo_lstm_pesos.weights.h5')

# Compilar (necesario para predicciones)
modelo.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall']
)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 CARGAR HISTORIAL DE ENTRENAMIENTO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import pickle

with open('modelos_lstm/historial_entrenamiento_lstm.pkl', 'rb') as file:
    historial = pickle.load(file)

# Acceder a métricas
# historial['loss']
# historial['val_loss']
# historial['accuracy']
# historial['val_accuracy']

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 EJEMPLO DE USO EN PRODUCCIÓN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import numpy as np
from tensorflow import keras
from joblib import load

# 1. Cargar modelo y escalador
modelo = keras.models.load_model('modelos_lstm/modelo_lstm_completo.keras')
scaler = load('modelos_lstm/scaler_lstm.bin')

# 2. Preparar nuevos datos (ejemplo)
# datos_motor_nuevo debe tener 30 ciclos × 14 características
# datos_motor_nuevo = np.array([...])  # Shape: (30, 14)

# 3. Normalizar (excluyendo columna 'motor' si existe)
# datos_normalizados = scaler.transform(datos_motor_nuevo)

# 4. Expandir dimensiones para batch
# datos_batch = np.expand_dims(datos_normalizados, axis=0)  # Shape: (1, 30, 14)

# 5. Predecir
# probabilidad_fallo = modelo.predict(datos_batch)[0][0]
# estado = 'FALLO INMINENTE' if probabilidad_fallo > 0.5 else 'NORMAL'

# print(f"Probabilidad de fallo: {probabilidad_fallo:.2%}")
# print(f"Estado del motor: {estado}")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 MÉTRICAS DEL MODELO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Accuracy:  0.9831 (98.31%)
• Precision: 0.8920 (89.20%)
• Recall:    0.9700 (97.00%)
• F1-Score:  0.9293 (92.93%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
