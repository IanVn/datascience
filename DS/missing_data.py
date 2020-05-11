import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Importando el dataset
data = pd.read_csv('Data/Data.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, 3].values

###################### Tratamiento de los valores faltantes
# Reemplazando los valores NaN con el objeto Imputer en donde los valores NaN tienen la media de los valores de las columnas
imputer = SimpleImputer( strategy='mean')
# Se devuelve una instancia de imputer, pasando como parametros los valores a calcular para reemplazar los valores NaN, los valores de X
# deben ser numeros para calcular la media
imputer = imputer.fit( X[:, 1:3] )

# Los valores faltantes del array X se reemplazan por los valores de la media calculados, el metodo devuelve un array de filas
# por ultimo los valores del array original se reemplazan por los valores del imputer
X[:, 1:3] = imputer.transform( X[:, 1:3] )
print( X[:, 1:3] )