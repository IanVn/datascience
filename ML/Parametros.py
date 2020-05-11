import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Definiendo los valores de X de acuerdo con un valor aleatorio conforme a una normal
x = 1.5 + 2.5*np.random.randn(100)

# Definiendo el residuo
res = 0 + 0.8*np.random.randn(100)

# Modelo de predicción es decir la ecuacion que modelará la regresión lineal
y_pred = 5 + 1.9 * x

# Modelo real con el valor del error
y_act = 5 + 1.9 * x + res

# Convertimos los valores a lista
x_list = x.tolist()
y_pred_list = y_pred.tolist()
y_act_list = y_act.tolist()

# Haciendo el dataframe, requerimos un diccionario con un conjunto de valores
data = pd.DataFrame({
    'x': x_list,
    'y_actual': y_act_list,
    'y_prediccion': y_pred_list
})

# Haciendo la media de los valores de la Y
y_mean = [ np.mean(data['y_actual']) for i in range(1, len(x_list) + 1) ]

# Obteniendo los valores
data['SSR'] = (data['y_prediccion'] - np.mean(data['y_actual']))**2
data['SSD'] = ( data['y_prediccion'] - data['y_actual'])**2
data['SST'] = ( data['y_actual'] - np.mean(data['y_actual']))**2
print(data.head())

# Sumas totales
SSR = sum(data['SSR'])
SSD = sum(data['SSD'])
SST = sum(data['SST'])

print(f'SSR: {SSR} , SSD: {SSD}, SST: {SST}')

# Encontrando los valores de los parametros de la regresion lineal
# Promedio de X
x_mean = np.mean(data['x'])
y_mean = np.mean(data['y_actual'])

# Haciendo las operaciones
data['beta_n'] = (data['x'] - x_mean) * (data['y_actual'] - y_mean )
data['beta_d'] = ( data['x'] - x_mean )**2

# Obteniendo beta
beta = sum(data['beta_n'])/sum(data['beta_d'])

# Obteniendo alfa
alfa = y_mean - beta * x_mean

# Obteniendo el modelo
data['y_model'] = beta * data['x'] + alfa

# Obteniendo los  parametros
SSR = sum( (data['y_model'] - y_mean ) ** 2 )
SSD = sum( ( data['y_model'] - data['y_actual'] )**2 )
SST = sum( (data['y_actual'] - y_mean ) **2 )
print(f'SSR: {SSR} , SSD: {SSD}, SST: {SST}')

y_mean = [ np.mean(data['y_actual']) for i in range(1, len(x_list) + 1) ]
# Imprimiendo las graficas
plt.plot(data['x'], data['y_prediccion'])
plt.plot(data['x'], data['y_actual'], 'ro')
plt.plot( data['x'], y_mean, 'g' )
plt.plot( data['x'], data['y_model'] )
plt.title('Valor actual vs predicción')
plt.show()

# Obteniendo el error estandar adicional
RSE = np.sqrt( SSD / (len(data)- 2) )
print(RSE)





