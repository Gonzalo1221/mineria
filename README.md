# Análisis y Predicción de Ventas con Python y Streamlit

## Descripción General

Aplicación web interactiva desarrollada con **Python** y **Streamlit** que permite analizar datos de ventas de un supermercado, generar visualizaciones complejas y entrenar un modelo de machine learning para predecir ventas futuras. El proyecto implementa técnicas de análisis exploratorio de datos (EDA) y modelado predictivo con interfaz amigable para usuarios.


## Objetivos del Proyecto

1. **Cargar y Limpiar Datos**: Importar dataset, eliminar duplicados y validar integridad
2. **Análisis Exploratorio (EDA)**: Generar 6 visualizaciones diferentes para explorar los datos
3. **Modelado Predictivo**: Entrenar modelo de regresión lineal para predicciones
4. **Interfaz Web**: Crear aplicación interactiva con Streamlit
5. **Evaluación del Modelo**: Calcular y mostrar métricas de desempeño


## Estructura del Proyecto

```
parcial/
├── app.py                          # Aplicación principal (151 líneas)
├── data/
│   └── Sample-Superstore.csv      # Dataset de ventas
├── notebooks/
│   └── analisis_exploratorio.ipynb # Análisis inicial
├── models/                         # Carpeta para modelos guardados
├── requirements.txt                # Dependencias
└── README.md                       # documentación
```


## Tecnologías Utilizadas

| Librería | Versión | Uso |
|----------|---------|-----|
| **Streamlit** | Latest | Framework web |
| **Pandas** | Latest | Manipulación de datos |
| **NumPy** | Latest | Operaciones numéricas |
| **Matplotlib** | Latest | Visualizaciones |
| **Seaborn** | Latest | Gráficos estadísticos |
| **Scikit-learn** | Latest | Machine Learning |
| **Python** | 3.7+ | Lenguaje base |


## Características Principales de la Aplicación

### **TAB 1: DATOS**
Muestra información general del dataset:
- Total de registros en el dataset
- Total de columnas
- Primeras 10 filas de datos
- Estadísticas descriptivas (media, mediana, desv. estándar, etc.)

### **TAB 2: ANÁLISIS EXPLORATORIO (EDA)** COMPLETO
Contiene 7 visualizaciones diferentes:

1. **Métricas en Columnas**
   - Ventas Totales (suma de todas las ventas)
   - Ganancias Totales (suma de ganancias)
   - Margen de Ganancia (% de rentabilidad)

2. **Ventas por Región**  Bar Chart
   - Comparación de ventas en diferentes regiones
   - Identificar regiones más vendedoras

3. **Ventas por Categoría**  Bar Chart
   - Desglose de ventas por categoría de producto
   - Identificar categorías más rentables

4. **Distribución por Segmento**  Pie Chart
   - Proporción de ventas por segmento de cliente
   - Colores diferenciados para cada segmento

5. **Distribución de Descuentos**  Histograma
   - Análisis de frecuencia de descuentos aplicados
   - Identificar rangos de descuento más comunes

6. **Top 5 Productos Rentables**  Bar Chart Horizontal
   - Los 5 productos que generan más ganancia
   - Ordenados de mayor a menor rentabilidad

7. **Ganancias por Región**  Bar Chart
   - Comparación de ganancias (no solo ventas)
   - Muestra rentabilidad real de cada región

8. **Matriz de Correlación**  Heatmap
   - Correlación entre variables numéricas: Sales, Quantity, Discount, Profit
   - Escala de colores (rojo=positivo, azul=negativo)
   - Ayuda a identificar relaciones entre variables

### **TAB 3: MODELO**
Modelado predictivo con regresión lineal:
- **Variables Independientes**: Quantity, Discount
- **Variable Dependiente**: Sales (Ventas)
- **Train-Test Split**: 80% entrenamiento, 20% prueba

**Métricas de Desempeño**:
- R² Score (Entrenamiento y Prueba)
- MAE: Error Absoluto Medio
- RMSE: Raíz del Error Cuadrático Medio

**Información del Modelo**:
- Tipo: Regresión Lineal (LinearRegression)
- Coeficientes mostrados para interpretación
- Precisión del modelo en dataset de prueba

### **TAB 4: PREDICCIONES**
Interfaz interactiva para hacer predicciones:
- **Slider 1**: Cantidad de items (1-14)
- **Slider 2**: Descuento aplicado (0-80%)
- **Botón**: "Predecir Venta"
- **Resultado**: Venta predicha en tiempo real


##  Instalación y Ejecución

### **Requisitos Previos**
- Python 3.7 o superior
- pip (gestor de paquetes de Python)
- Git (opcional, para clonar el repositorio)

### **Paso 1: Clonar o Descargar el Repositorio**
```bash
git clone <URL_DEL_REPOSITORIO>
cd proyectoventas
```

### **Paso 2: Crear Entorno Virtual (Recomendado)**

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**En macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### **Paso 3: Instalar Dependencias**
```bash
pip install -r requirements.txt
```

### **Paso 4: Ejecutar la Aplicación**
```bash
streamlit run app.py
```

### **Paso 5: Acceder a la Aplicación**
La aplicación se abrirá automáticamente en:
```
http://localhost:8501
```

Si no se abre automáticamente, copia y pega la URL en tu navegador.


##  Flujo de Trabajo Completo

```
1. CARGA DE DATOS
   └─ Leer CSV con encoding latin
   └─ Validar estructura
   └─ Eliminar duplicados

2. ANÁLISIS EXPLORATORIO (EDA)
   └─ Estadísticas descriptivas
   └─ 7 visualizaciones diferentes
   └─ Análisis de correlaciones

3. PREPARACIÓN PARA MODELADO
   └─ Seleccionar features (Quantity, Discount)
   └─ Dividir datos (80% train, 20% test)
   └─ Crear objeto de modelo

4. ENTRENAMIENTO DEL MODELO
   └─ Ajustar modelo con datos de entrenamiento
   └─ Calcular predicciones
   └─ Evaluar con métricas

5. PREDICCIONES INTERACTIVAS
   └─ Usuario ingresa valores
   └─ Modelo predice resultado
   └─ Mostrar predicción en interfaz
```


##  Dataset: Sample-Superstore

**Descripción**: Datos de ventas reales de una tienda comercial con múltiples regiones, categorías y segmentos de clientes.

**Columnas Principales**:
- **Sales**: Monto de ventas (variable objetivo)
- **Quantity**: Cantidad de unidades vendidas
- **Discount**: Porcentaje de descuento aplicado
- **Profit**: Ganancia generada
- **Region**: Región geográfica (West, East, Central, South)
- **Category**: Categoría de producto (Furniture, Office Supplies, Technology)
- **Segment**: Segmento de cliente (Consumer, Corporate, Home Office)
- **Product_ID**: Identificador único del producto


##  Métricas de Evaluación del Modelo

### **R² (Coeficiente de Determinación)**
- Rango: 0 a 1
- Significado: Porcentaje de varianza explicada por el modelo
- Interpretación: Más cercano a 1 = mejor ajuste

### **MAE (Error Absoluto Medio)**
- Rango: 0 a infinito
- Significado: Promedio de errores absolutos
- Interpretación: Menor valor = mejor rendimiento

### **RMSE (Raíz del Error Cuadrático Medio)**
- Rango: 0 a infinito
- Significado: Penaliza más los errores grandes
- Interpretación: Menor valor = mejor rendimiento


##  Archivo requirements.txt

Contiene todas las dependencias necesarias:
```
streamlit>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```


##  Configuración de la Aplicación

La aplicación está configurada en `app.py` con:
- **Página**: Wide layout (ancho completo)
- **Título**: "Análisis y Predicción de Ventas"
- **Descripción**: Markdown con explicación del proyecto
- **Cache**: Habilitado para mejor rendimiento
- **Codificación**: UTF-8 para caracteres especiales


**Repositorio**: https://github.com/Gonzalo1221/mineria


##  Troubleshooting (Solución de Problemas)

### Problema: "No se encuentra el archivo CSV"
**Solución**: Asegúrate de que el archivo `Sample-Superstore.csv` esté en la carpeta `data/`

### Problema: "ModuleNotFoundError"
**Solución**: Ejecuta `pip install -r requirements.txt` para instalar todas las librerías

### Problema: "La aplicación es muy lenta"
**Solución**: La caché está habilitada. Intenta refrescar la página o limpiar caché del navegador

### Problema: "Los gráficos no se muestran"
**Solución**: Verifica que Matplotlib y Seaborn estén instalados: `pip install matplotlib seaborn`


##  Notas Importantes

- El modelo usa solo 2 variables (Quantity, Discount) para mantener simplicity
- Los datos se cargan en caché para mejor rendimiento
- Todas las visualizaciones se generan en tiempo real
- La interfaz es completamente responsiva
- El archivo `.gitignore` excluye archivos innecesarios


##  Soporte

Para preguntas o problemas:
1. Revisa este README
2. Verifica que todas las dependencias estén instaladas
3. Intenta ejecutar nuevamente el comando `streamlit run app.py`
4. Contacta a los integrantes del grupo