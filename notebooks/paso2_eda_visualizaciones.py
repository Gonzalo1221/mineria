# PASO 2: ANALISIS EXPLORATORIO (EDA) - VISUALIZACIONES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Cargar dataset
df = pd.read_csv('C:/Users/Gonzalo01/Documents/Downloads/PPP/parcial/data/Sample-Superstore.csv', encoding='latin1')
df_limpio = df.drop_duplicates()

print('\n1. VISUALIZANDO VENTAS POR REGION')
print('='*70)
ventas_region = df_limpio.groupby('Region')['Sales'].sum().sort_values(ascending=False)
print(ventas_region)
print(f'\nTotal de ventas: ${ventas_region.sum():,.2f}')

print('\n2. VENTAS POR CATEGORIA')
print('='*70)
ventas_categoria = df_limpio.groupby('Category')['Sales'].sum().sort_values(ascending=False)
print(ventas_categoria)
print('\n3. CANTIDAD DE PEDIDOS POR SEGMENTO')
print('='*70)
pedidos_segmento = df_limpio.groupby('Segment').size()
print(pedidos_segmento)

print('\n4. ANALISIS DE MARGENES Y DESCUENTOS')
print('='*70)
print(f'Descuento promedio: {df_limpio["Discount"].mean():.2%}')
print(f'Margen promedio: {df_limpio["Profit"].sum() / df_limpio["Sales"].sum():.2%}')
print(f'Ganancia total: ${df_limpio["Profit"].sum():,.2f}')
print(f'Cantidad de pedidas perdidas: {(df_limpio["Profit"] < 0).sum()}')

print('\n5. CORRELACION ENTRE VARIABLES NUMERICAS')
print('='*70)
cols_numericas = ['Sales', 'Quantity', 'Discount', 'Profit']
correlacion = df_limpio[cols_numericas].corr()
print('\nCorrelacion de Profit con otras variables:')
print(correlacion['Profit'].sort_values(ascending=False))

print('\n6. PRODUCTOS MAS Y MENOS RENTABLES')
print('='*70)
rentabilidad_producto = df_limpio.groupby('Sub-Category')['Profit'].sum().sort_values(ascending=False)
print('\nTop 5 mas rentables:')
print(rentabilidad_producto.head())
print('\nTop 5 menos rentables:')
print(rentabilidad_producto.tail())

print('\n' + '='*70)
print('RESUMEN DEL ANALISIS EXPLORATORIO')
print('='*70)
print(f'Total de registros analizados: {len(df_limpio):,}')
print(f'Ventas totales: ${df_limpio["Sales"].sum():,.2f}')
print(f'Ganancia neta: ${df_limpio["Profit"].sum():,.2f}')
print(f'Margen de ganancia: {(df_limpio["Profit"].sum() / df_limpio["Sales"].sum()):.2%}')
print('\n✓ STEP 2 COMPLETADO - EDA REALIZADO CON EXITO')
