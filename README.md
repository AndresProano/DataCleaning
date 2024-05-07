# DataCleaning

<p>Use el conjunto de datos de los taxis New York para hacer las siguientes actividades:

- Limpie los datos: busque problemas con los datos para corregirlos
- Modifique los datos: use datos existentes para crear datos que pueden ser mejores
- Explore los datos: muestre cosas interesantes acerca de los datos y presente sus conclusiones usando diferentes figuras.
- Corra un modelo simple como regresión lineal y muestre los resultados . 

Datos:</p>

[nyc_taxi_hw .csv](https://github.com/AndresProano/DataCleaning/blob/main/nyc_taxi_hw%20.csv)

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

### Leer los datos

<p>Crearemos una variable "data" donde leeremos el archivo .csv donde se encuentran los datos</p>

````
data = pd.read_csv('/Users/andres/Downloads/nyc_taxi_hw.csv')
````

### Primera visualización de datos

<p>Para tener una primera visualización de datos ocupamos las siguientes dos funciones: </p>

````
data.head()
````

![Figura 1. Head inicial](https://github.com/AndresProano/DataCleaning/blob/main/images/1.png)

````
data.info()
````

![Figura 2. Info inicial](https://github.com/AndresProano/DataCleaning/blob/main/images/2.png)

### Limpieza de datos

<p>Basándonos en esta primera observación de datos podemos analizar lo siguiente: 

"key" es un conjunto de año, mes, día de la semana y hora que está reflejado como un dato "object". Podemos realizar la transformación de estos datos a "datetime64[ns]" para utilizarlos de una mejor manera. Para esto, utilizamos los comandos:</p>

````
data['key'] = pd.to_datetime(data['key'])
data['key'].iloc[0]
````

````
data["hour"] = data["key"].dt.hour
data["day_of_week"] = data["key"].dt.dayofweek
data["month"] = data["key"].dt.month
data["year"] = data["key"].dt.year
````

<p>"pickup_datetime" cuenta con la misma propiedad de un conjunto de datos que describen una fecha, en este caso, el momento en que se recogió al pasajero. Por lo que, podemos hacer el mismo procedimiento</p>

````
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
data['pickup_datetime'].iloc[0]
````

````
data["hour"] = data["pickup_datetime"].dt.hour
data["day_of_week"] = data["pickup_datetime"].dt.dayofweek
data["month"] = data["pickup_datetime"].dt.month
data["year"] = data["pickup_datetime"].dt.year
````

![Figura 3. Info después de transformación de key y pickup_datetime ]()

<p>Ahora, podemos visualizar que estos dos datos son lo mismo, por lo que podemos eliminar uno de esto. En este caso, vamos a eliminar "pickup_datetime"</p>

````
data.drop(['pickup_datetime'], axis=1, inplace=True)
````

<p>Resultado hasta el momento:</p>

![Figura 4. Info después de eliminar pickup datetime]()




