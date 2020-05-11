import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Leemos el csv
data = pd.read_csv('./datasets/ads/Advertising.csv')

# generamos unos valores aleatorios
a = np.random.randn( len(data) )

# Filtramos los datos tal que los valores aleatorios sean menor a 0.8, la variable check
# devuelve un array  de booleanos en el que cada posicion es true si el elemento en dicha posicion
# cumple con la condicion
check = ( a < 0.8 )
# Como el array check tiene booleanos cada valor booleano representa una fila, al poner estos valores dentro
# del array del dataset filtrará aquellas filas que tengan true, esto es un subconjunto del dataset
training = data[check]

# Ahora se niegan los valores del array de check, los valores true son false y lo mismo, solo se filtran aquellos
# filas que tengan true, es decir, que en el primer elemento si es true se dejará la primera fila, etc
test = data[~check]
# Calculando el modelo
lm = smf.ols('Sales~TV+Radio', data=training).fit()
print(lm.summary())

# Valores de prediccion
sales_pred = lm.predict(test)
print(sales_pred)

# SSD
SSD = sum( (test['Sales'] - sales_pred) ** 2 )
RSE = np.sqrt( SSD / (len(test)- 3) )

# Media
sales_mean = np.mean( test['Sales'] )
error = RSE / sales_mean
print(f'Error:  {error}')



