import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing


# Importando el dataset
data = pd.read_csv('Data/Data.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, 3].values

############################ Codificacion de las variables categoricas #####################################
# Codificacion de variables categoricas a variables discretas
# Creamos una instancia de label encoder
labelencoder_X = preprocessing.LabelEncoder()
# Usamos el metodo fit_transform para devolver las etiquetas codificadas
X[:,0] = labelencoder_X.fit_transform( X[:, 0] )

# Usamos variables dummy
# Con el parametro categories se busca automaticamente cuales columnas se van a categorizar, con el valor [0]
# decimos que de los datos de entrenamiento se categorice la columna 0
ct = ColumnTransformer( [ ('one_hot_encoder', OneHotEncoder( categories='auto' ), [0]) ], remainder='passthrough')
# El metodo fit recibe los datos para ejecutar, se localiza la columna 0 y se categorizan las variables con un valor binario, esto devuelve
# los datos con las columnas categorizadas agregadas, despues convertimos ese array a un array de numpy
X = np.array( ct.fit_transform( X ))

# Codificar los valores de Y
labelencoder_y = preprocessing.LabelEncoder()
y = labelencoder_y.fit_transform(Y)