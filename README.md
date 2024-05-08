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
from geopy.distance import geodesic
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error
````

<p> Utilizaremos 
  
- Pandas: librería de manipulación y análisis de datos que proporciona estructuras de datos flexibles y herramientas para trabajar con ellas.
- Numpy: librería fundamental para la computación numérica en Python. Proporciona una matriz multidimensional eficiente, objetos de matriz y una variedad de funciones matemáticas para trabajar con estos arrays.
- Matplotlib.pyplot: librería de visualización de datos en Python.
- Seaborn: librería de visualización de datos en Python, construida sobre Matplotlib.
- Geopy.distance import geodesic: Es utilizado para calcular la distancia geodésica entre dos puntos dados en la Tierra utilizando coordenadas geográficas (latitud y longitud)
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

![Figura 3. Info después de transformación de key y pickup_datetime ](https://github.com/AndresProano/DataCleaning/blob/main/images/3.png)

<p>Ahora, podemos visualizar que estos dos datos son lo mismo, por lo que podemos eliminar uno de esto. En este caso, vamos a eliminar "pickup_datetime"</p>

````
data.drop(['pickup_datetime'], axis=1, inplace=True)
````

<p>Resultado hasta el momento:</p>

![Figura 4. Info después de eliminar pickup datetime](https://github.com/AndresProano/DataCleaning/blob/main/images/4.png)

<p>Otro punto importante que podemos notar dentro de los datos es que "Unamed:0" puede interpretarse como un identificador de los datos. Pero, también contamos con "key" que vendría a ser un identificador de igual forma. Podemos eliminar "Unnamed:0". </p>

````
data.drop(['Unnamed: 0'], axis=1, inplace=True)
````

![Figura 5. Info después de eliminar Unnamed: 0](https://github.com/AndresProano/DataCleaning/blob/main/images/5.png)

<p>A manera de obtener mayor información de los datos, podemos crear una función que nos retorne si el viaje se realizó en la mañana, tarde, noche. Para esto colocamos: </p>

````
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

data['time_of_day'] = data['hour'].apply(get_time_of_day)
````

![Figura 6. Info después de agregar tiempo de dia](https://github.com/AndresProano/DataCleaning/blob/main/images/6.png)

![Figura 7. Data despues de agregar tiempo de dia](https://github.com/AndresProano/DataCleaning/blob/main/images/7.png)

<p>Otra acción que podemos realizar es el cambio de "passenger_count" de int64 a int32 debido a que no se utiliza de forma eficiente el int64.</p>

````
data['passenger_count'] = data['passenger_count'].astype('int32')
````

![Figura 8. Info despues de transformar passegner_count a int32](https://github.com/AndresProano/DataCleaning/blob/main/images/8.png)

<p>Ahora, vamos a verificar los valores de pasajeros que sean igual a cero. Estos no tienen sentido dentro del análisis de datos por lo que también los eliminaremos.</p>

````
datos_pasajeros_cero = data[data['passenger_count'] == 0]
data = data[data['passenger_count'] != 0]
````

<p>Para verificar que fueron eliminados utilizaremos</p>

````
datos_pasajeros_cero.head()
````

![Figura 15. DatosPasajeroCero](https://github.com/AndresProano/DataCleaning/blob/main/images/15.png)

<p>También verificaremos los precios que estén en cero ya que no tienen sentido dentro del análisis. Serán eliminados.</p>

````
data = data[data['fare_amount'] != 0]
````

<P>Para verificar cuantos datos fueron eliminados utilizaremos</P>

````
data.info()
````

![Figura 16. Info despues de eliminar fareamount](https://github.com/AndresProano/DataCleaning/blob/main/images/16.png)

<p>Podemos visualizar una disminución de datos, por lo que concluimos que están eliminados</p>

<p>Ahora, en la misma categoria de fare_amount eliminaremos los precios que sean menores a cero, ya que no tiene sentido dentro del análisis</p>

````
datos_precio_menos = data[data['fare_amount'] <= 0]
````

<p>Eliminaremos las localizaciones que tengan como coordenadas 0.0, esto quiere decir que en pickup_longitude, pickup_latitude, dropoff_longitude y dropoff_latitude filtraremos aquellas que se encuentren con 0 ya que no toman lugar dentro de Nueva York</p>

````
valores_cero = data[(data['pickup_longitude'] == 0) | (data['pickup_latitude'] == 0) | (data['dropoff_longitude'] == 0) | (data['dropoff_latitude'] == 0)]
data = data[(data['pickup_longitude'] != 0) & (data['pickup_latitude'] != 0) & (data['dropoff_longitude'] != 0) & (data['dropoff_latitude'] != 0)]
````

![Figura 17. Info despues de eliminar valores cero](https://github.com/AndresProano/DataCleaning/blob/main/images/17.png)

<p>Entre los datos contamos con "pickup_longitude", "pickup_latitude", "dropoff_longitude" y "dropoff_latitude". Con estos datos, podemos calcular la distancia que se recorrió aproximadamente en cada viaje. Para realizar esto, utilizamos</p>

````
pickup_coords = data[['pickup_latitude', 'pickup_longitude']]
dropoff_coords = data[['dropoff_latitude', 'dropoff_longitude']]

distances = []
for pickup, dropoff in zip(pickup_coords.values, dropoff_coords.values):
    try:
        distance = geodesic((pickup[0], pickup[1]), (dropoff[0], dropoff[1])).kilometers
        distances.append(distance)
    except ValueError:
        distances.append(None)

data['distance_km'] = distances
````

![Figura 9. Info despues de agregar distancia en KM](https://github.com/AndresProano/DataCleaning/blob/main/images/km.png)

<p>Podemos visualizar y eliminar los datos que cuenten con km cero, debido a que no son posibles </p>

````
data = data[data['distance_km'] != 0]
````

![Figura 18. Info despues de eliminar km de cero](https://github.com/AndresProano/DataCleaning/blob/main/images/18.png)

<p>Ahora podemos verificar cualquier tipo de valores atípicos de km con la siguiente función</p>

````
print(data['distance_km'].describe())

Q1 = data['distance_km'].quantile(0.25)
Q3 = data['distance_km'].quantile(0.75)
IQR = Q3 - Q1
filtro_atipico = (data['distance_km'] < (Q1 - 1.5 * IQR)) | (data['distance_km'] > (Q3 + 1.5 * IQR))
valores_atipicos = data[filtro_atipico]
print("Valores atípicos:")
print(valores_atipicos)
````

![Figura 19. valores atipicos](https://github.com/AndresProano/DataCleaning/blob/main/images/19.png)

<p>Para verificar que no tenemos datos redundantes, podemos usar:</p>

````
numeric_df = data.select_dtypes(include=['number'])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=True, linewidth=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
````

![Figura 10. correlation matrix heatmap](https://github.com/AndresProano/DataCleaning/blob/main/images/9.png)

<p>Podemos observar que no contamos con datos redundantes, por lo que podríamos dejar los datos como se encuentran y empezar con el análisis.</p>

### Análisis de datos

<p>Vamos a realizar un conteo de viajes dependiendo de diferentes factores de la siguiente manera: </p>

````
def plot_count_by_category(data, category):
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=category,
        y='count',
        data=data.groupby(category).size().reset_index(name='count'),
        palette='viridis'
    )
    plt.xlabel(category.capitalize())
    plt.ylabel('Count')
    plt.title(f'Count of Trips by {category.capitalize()}')
    plt.show()
````

<p>Como primera comparación, lo vamos a hacer con la hora del día.</p>

````
plot_count_by_category(data, 'hour')
````

![Figura 11. trips by hora](https://github.com/AndresProano/DataCleaning/blob/main/images/10.png)

<p>En esta imagen podemos observar que existe una mayor cantidad de viajes entre las 18:00 y 19:00, esto puede deberse a que son las horas pico dentro de la ciudad de Nueva York. Esto podemos asociarlo con una mayor cantidad de tráfico debido a la demanda. 
Así mismo, podemos observar que de 1:00 a 6:00 la cantidad de viajes es mucho menor en comparación al resto del día debido a que la mayoría de gente descansa
Existe una cantidad similar de viajes entre las 7:00 y 17:00.
En base al gráfico podemos observar que la mayor cantidad de viajes se realizan entre las 18:00 y las 23:00.</p>

<p>Para tener una visión más general de la cantidad de viajes que se realizan entre la mañana, tarde y noche podemos utilizar el periodo del día</p>

````
plot_count_by_category(data, 'time_of_day')
````

![Figura 12. trips by tiempo de dia](https://github.com/AndresProano/DataCleaning/blob/main/images/11.png)

<p>Como se describió antes, podemos observar que la mayor cantidad de viajes se realizan en la noche. Posteriormente tendremos que la mayor cantidad de viajes se realiza en la mañana seguido por la tarde y finalmente la noche. </p>

<p>Ahora realizaremos una comparación con el día de la semana</p>

````
plot_count_by_category(data, 'day_of_week')
````

![Figura 13. trips by day](https://github.com/AndresProano/DataCleaning/blob/main/images/12.png)

<p>Asumiendo que 0 representa lunes y 6 representa domingo, tenemos que los días donde más viajes existen son viernes y sábado. Esto puede ser debido a que es considerado el fin de semana por lo que se puede llegar a aprovechar en visitas a parques o atracciones turísticas, pero también podemos considerar un indicador de mayor flujo de personas dentro de la ciudad lo que deriba en tráfico. 
Los días con menos viajes vendrían a ser lunes y domingo, que podríamos relacionarlo con el inicio de semana por lo que la gente se ve en la necesidad de volver a sus rutinas y tratar de evitar el tráfico usando otro tipo de transportes.</p>

<p>Ahora realizaremos una comparación con el mes</p>

````
plot_count_by_category(data, 'month')
````

<p>En esta gráfica podemos observar que de enero a junio existe un flujo casi constante de viajes realizados. Esto podría deberse a factores como:
  
- El clima: El clima podría jugar un papel fundamental en que las personas prefieran tomar taxi a caminar
- Eventos turísticos: Se conoce que New York es una de las ciudades más transitadas por extranjeros, por lo que las fechas de mayor flujo de viajes coincide con ciertas festividades de latinoamérica y europa
- Desplazamientos laborales: Al ser la primera parte del año, las personas se encuentran viajando hacia sus trabajos y podrían llegar a preferir el uso de taxis.</p>

<p>Ahora realizaremos una comparación con el año</p>

````
plot_count_by_category(data, 'year')
````

![Figura 14. trips by year](https://github.com/AndresProano/DataCleaning/blob/main/images/14.png)

<p>En este gráfico podemos observar que en la mayoría de años existe un número similar de viajes. El último año, 2015, va en tendencia a tener la misma cantidad de viajes ya que hasta el momento solo existen datos hasta 2015-06-30.</p>


