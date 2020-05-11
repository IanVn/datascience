import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Importando el dataset
data = pd.read_csv('Data/Data.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, 3].values

########################################### Tratamiento de valores faltantes ##############################################################

# Reemplazando los valores NaN con el objeto Imputer en donde los valores NaN tienen la media de los valores de las columnas
# imputer = SimpleImputer( strategy='mean')
# Se devuelve una instancia de imputer, pasando como parametros los valores a calcular para reemplazar los valores NaN, los valores de X
# deben ser numeros para calcular la media
# imputer = imputer.fit( X[:, 1:3] )

# Los valores faltantes del array X se reemplazan por los valores de la media calculados, el metodo devuelve un array de filas
# por ultimo los valores del array original se reemplazan por los valores del imputer
# X[:, 1:3] = imputer.transform( X[:, 1:3] )
# print( X[:, 1:3] )


################################################### Codificacion de variables categoricas ##########################################################

# Codificacion de variables categoricas a variables discretas
# Creamos una instancia de label encoder
# labelencoder_X = preprocessing.LabelEncoder()
# Usamos el metodo fit_transform para devolver las etiquetas codificadas
# X[:,0] = labelencoder_X.fit_transform( X[:, 0] )

# Usamos variables dummy
# Con el parametro categories se busca automaticamente cuales columnas se van a categorizar, con el valor [0]
# decimos que de los datos de entrenamiento se categorice la columna 0
# ct = ColumnTransformer( [ ('one_hot_encoder', OneHotEncoder( categories='auto' ), [0]) ], remainder='passthrough')
# El metodo fit recibe los datos para ejecutar, se localiza la columna 0 y se categorizan las variables con un valor binario, esto devuelve
# los datos con las columnas categorizadas agregadas, despues convertimos ese array a un array de numpy
# X = np.array( ct.fit_transform( X ))

# Codificar los valores de Y
#labelencoder_y = preprocessing.LabelEncoder()
#y = labelencoder_y.fit_transform(Y)


####################################### Division del dataset en datos de entrenamiento y de testing #################################################
# Division del dataset en un conjunto de entrenamiento y de testing, el parametro random_state es para garantizar que siempre se obtendrá
# los mismos subconjuntos de testing y entrenamiento
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0 )
print('Xtrain',X_train)
print('X_test',X_test)
print('y_train',y_train)
print('y_test',y_test)


########################################## Escalamiento de valores ######################################################################
# Escalamos los valores de entrenamiento y de testing
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform( X_train )
# X_test = sc_X.transform( X_test )
# print(X_train)
# print(X_test)







