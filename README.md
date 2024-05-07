# DataCleaning

<p>Use el conjunto de datos de los taxis New York para hacer las siguientes actividades:

- Limpie los datos: busque problemas con los datos para corregirlos
- Modifique los datos: use datos existentes para crear datos que pueden ser mejores
- Explore los datos: muestre cosas interesantes acerca de los datos y presente sus conclusiones usando diferentes figuras.
- Corra un modelo simple como regresión lineal y muestre los resultados . 

Datos:</p>

<p>
[nyc_taxi_hw .csv](https://github.com/AndresProano/DataCleaning/blob/main/nyc_taxi_hw%20.csv)
</p>

## Procedimiento: 

### Importar librerias

<p>Primero, necesitaremos importar las siguientes librerías que se utilizarán para realizar el análisis de datos</p>

````
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error
````

<p> Utilizaremos 
  
- Pandas: librería de manipulación y análisis de datos que proporciona estructuras de datos flexibles y herramientas para trabajar con ellas.
- Numpy: librería fundamental para la computación numérica en Python. Proporciona una matriz multidimensional eficiente, objetos de matriz y una variedad de funciones matemáticas para trabajar con estos arrays.
- Matplotlib.pyplot: librería de visualización de datos en Python.
- Seaborn: librería de visualización de datos en Python, construida sobre Matplotlib.
- Sklearn.preprocessing import OneHotEncoder: librería de aprendizaje automático en Python que proporciona herramientas simples y eficientes para análisis predictivo y minería de datos.
- Sklearn.linear_model import LinearRegression: Esta clase de Scikit-learn se utiliza para implementar el modelo de regresión lineal.
- Sklearn.metrics import mean_squared_error: Esta función de Scikit-learn se utiliza para calcular el error cuadrático medio (MSE), que es una medida de la calidad de un estimador en términos del promedio de los errores al cuadrado.
- Sklearn.metrics import r2_score: Esta función de Scikit-learn se utiliza para calcular el coeficiente de determinación (R cuadrado), que es una medida de qué tan bien el modelo se ajusta a los datos.
- Sklearn.metrics import mean_absolute_error: Esta función de Scikit-learn se utiliza para calcular el error absoluto medio (MAE), que es una medida de la diferencia absoluta entre las predicciones y los valores reales.
- Sklearn.metrics import mean_squared_log_error: Esta función de Scikit-learn se utiliza para calcular el error cuadrático medio logarítmico (MSLE), que es una métrica de evaluación de modelos para problemas de regresión donde las predicciones y los objetivos se interpretan como logaritmos de valor real.</p>
