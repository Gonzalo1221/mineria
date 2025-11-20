# PASO 3: MODELADO PREDICTIVO - MACHINE LEARNING

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print('\n1. CARGANDO Y PREPARANDO DATOS')
print('='*70)
df = pd.read_csv('C:/Users/Gonzalo01/Documents/Downloads/PPP/parcial/data/Sample-Superstore.csv', encoding='latin1')
df = df.drop_duplicates()
print(f'Dataset: {df.shape}')

# Seleccionar features numericas
X = df[['Quantity', 'Discount']].copy()
y = df['Sales'].copy()
print(f'Features: {X.shape}, Target: {y.shape}')

print('\n2. DIVIDIENDO DATASET EN ENTRENAMIENTO Y PRUEBA')
print('='*70)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Entrenamiento: {X_train.shape}')
print(f'Prueba: {X_test.shape}')

print('\n3. ENTRENANDO MODELO DE REGRESION LINEAL')
print('='*70)
modelo = LinearRegression()
modelo.fit(X_train, y_train)
print('Modelo entrenado exitosamente')
print(f'Coeficientes: {modelo.coef_}')
print(f'Intercepto: {modelo.intercept_:.2f}')

print('\n4. HACIENDO PREDICCIONES')
print('='*70)
y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)
print('Predicciones realizadas')
print(f'Primeras 5 predicciones: {y_pred_test[:5]}')

print('\n5. EVALUANDO MODELO CON METRICAS')
print('='*70)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print('\nMETRICAS DE DESEMPENO:')
print(f'R² Entrenamiento: {r2_train:.4f}')
print(f'R² Prueba: {r2_test:.4f}')
print(f'MSE Prueba: {mse_test:.4f}')
print(f'MAE Prueba: {mae_test:.4f}')
print(f'RMSE Prueba: {np.sqrt(mse_test):.4f}')

print('\n' + '='*70)
print('RESUMEN - MODELO PREDICTIVO')
print('='*70)
print(f'Modelo: Regresión Lineal')
print(f'Variables independientes: Quantity, Discount')
print(f'Variable objetivo: Sales')
print(f'Precision (R²): {r2_test:.2%}')
print('\n✓ STEP 3 COMPLETADO - MODELO ENTRENADO')
