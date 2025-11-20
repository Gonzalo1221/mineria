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
    df = pd.read_csv('C:/Users/Gonzalo01/Documents/Downloads/PPP/parcial/data/Sample-Superstore.csv', encoding='latin1')
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
