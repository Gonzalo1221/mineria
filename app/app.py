import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configurar página
st.set_page_config(page_title='Analisis Ventas', layout='wide')
st.title('📊 Analisis y Prediccion de Ventas')
st.markdown('Aplicacion para analizar datos de ventas y hacer predicciones')

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv('./data/Sample-Superstore.csv', encoding='latin1')
    return df.drop_duplicates()

df = load_data()

# Crear pestañas
tab1, tab2, tab3, tab4 = st.tabs(['Datos', 'EDA', 'Modelo', 'Predicciones'])

# TAB 1: DATOS
with tab1:
    st.header('Vista de Datos')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Total de Registros', len(df))
    with col2:
        st.metric('Total de Columnas', len(df.columns))
    st.write(df.head(10))
    st.subheader('Estadisticas Descriptivas')
    st.write(df.describe())

# TAB 2: EDA
with tab2:
    st.header('Analisis Exploratorio')
    col1, col2, col3 = st.columns(3)
    with col1:
        total_ventas = df['Sales'].sum()
        st.metric('Ventas Totales', f'${total_ventas:,.0f}')
    with col2:
        ganancia = df['Profit'].sum()
        st.metric('Ganancia Total', f'${ganancia:,.0f}')
    with col3:
        margen = (ganancia / total_ventas) * 100
        st.metric('Margen %', f'{margen:.1f}%')
    st.subheader('Ventas por Region')
    ventas_region = df.groupby('Region')['Sales'].sum()
    st.bar_chart(ventas_region)

    # Análisis por Categoría
st.subheader('Ventas por Categoría')
ventas_categoria = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
ventas_categoria.plot(kind='bar', color='steelblue', ax=ax)
ax.set_title('Ventas por Categoría de Producto')
ax.set_xlabel('Categoría')
ax.set_ylabel('Ventas ($)')
plt.xticks(rotation=45)
st.pyplot(fig)

# Análisis por Segmento de Cliente
st.subheader('Distribución de Ventas por Segmento')
ventas_segmento = df.groupby('Segment')['Sales'].sum()
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#FF9999', '#66B2FF', '#99FF99']
wedges, texts, autotexts = ax.pie(ventas_segmento, labels=ventas_segmento.index, autopct='%1.1f%%', colors=colors, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_weight('bold')
ax.set_title('Proporción de Ventas por Segmento')
st.pyplot(fig)

# Distribución de Descuentos
st.subheader('Distribución de Descuentos Aplicados')
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['Discount'] * 100, bins=20, color='coral', edgecolor='black')
ax.set_title('Histograma de Descuentos')
ax.set_xlabel('Descuento (%)')
ax.set_ylabel('Frecuencia')
st.pyplot(fig)

# Top 5 Productos Más Rentables
st.subheader('Top 5 Productos Más Rentables')
top_productos = df.groupby('Product ID')['Profit'].sum().nlargest(5)
fig, ax = plt.subplots(figsize=(10, 5))
top_productos.plot(kind='barh', color='green', ax=ax)
ax.set_title('Top 5 Productos Más Rentables')
ax.set_xlabel('Ganancia ($)')
st.pyplot(fig)

# Ganancias por Región
st.subheader('Ganancias por Región')
ganancias_region = df.groupby('Region')['Profit'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
ganancias_region.plot(kind='bar', color='purple', ax=ax)
ax.set_title('Ganancias Totales por Región')
ax.set_xlabel('Región')
ax.set_ylabel('Ganancia ($)')
plt.xticks(rotation=45)
st.pyplot(fig)

# Matriz de Correlación
st.subheader('Matriz de Correlación de Variables Numéricas')
df_numeric = df[['Sales', 'Quantity', 'Discount', 'Profit']]
corr_matrix = df_numeric.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, ax=ax, cbar_kws={'label': 'Correlación'})
ax.set_title('Correlación entre Variables')
st.pyplot(fig)

# TAB 3: MODELO
with tab3:
    st.header('Modelo de Regresion Lineal')
    X = df[['Quantity', 'Discount']].copy()
    y = df['Sales'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred_test = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('R2 Score', f'{r2:.4f}')
    with col2:
        st.metric('MAE', f'{mae:.2f}')
    with col3:
        st.metric('Coef. Quantity', f'{modelo.coef_[0]:.2f}')
    st.info(f'Modelo entrenado con exito. R²: {r2:.2%}')

# TAB 4: PREDICCIONES
with tab4:
    st.header('Hacer Predicciones')
    st.write('Ingresa los valores para obtener una prediccion:')
    col1, col2 = st.columns(2)
    with col1:
        quantity = st.slider('Cantidad', 1, 14, 5)
    with col2:
        discount = st.slider('Descuento (%)', 0, 80, 20) / 100
    if st.button('Predecir Venta'):
        X_pred = np.array([[quantity, discount]])
        prediction = modelo.predict(X_pred)[0]
        st.success(f'Venta Predicha: ${prediction:,.2f}')
