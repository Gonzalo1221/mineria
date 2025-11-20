# PASO 1: CARGA Y EXPLORACION DE DATOS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print('\n1. CARGANDO DATASET CON PANDAS')
print('='*70)

# Cargar el dataset
df = pd.read_csv('C:/Users/Gonzalo01/Documents/Downloads/PPP/parcial/data/Sample-Superstore.csv', encoding='latin1')
print(f'Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas')

# Ver las primeras filas
print('\nPrimeras 5 filas:')
print(df.head())

print('\n2. INFORMACION DEL DATASET')
print('='*70)
df.info()

print('\n3. ESTADISTICAS DESCRIPTIVAS')
print('='*70)
print(df.describe())

print('\n4. ANALISIS DE VALORES NULOS')
print('='*70)
nulos = df.isnull().sum()
print(nulos)

print('\n5. ANALISIS DE DUPLICADOS')
print('='*70)
duplicados = df.duplicated().sum()
print(f'Filas duplicadas: {duplicados}')

print('\n6. LIMPIEZA DE DATOS')
print('='*70)
df_limpio = df.drop_duplicates()
print(f'Dataset limpio: {df_limpio.shape[0]} filas, {df_limpio.shape[1]} columnas')
print(f'Filas eliminadas: {df.shape[0] - df_limpio.shape[0]}')

print('\n7. ANALISIS DE COLUMNAS')
print('='*70)
print('\nColumnas del dataset:')
for col in df_limpio.columns:
        print(f'  - {col}: {df_limpio[col].dtype}')

        print('\n' + '='*70)
        print('✓ PASO 1 COMPLETADO EXITOSAMENTE')
        print('='*70)
        