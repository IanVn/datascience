import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Leyendo el csv
data = pd.read_csv('./datasets/ads/Advertising.csv')

# Por medio de relacion se hace la estimacion en donde el parametro formular o primer parametro es la relacion de la variable de salida
# con la variable predictora, el segundo argumento es el dataframe y con el metodo fit se hace el ajuste completo
model = smf.ols('Sales~TV', data=data).fit()
# EL valor de intercept es el valor alfa y el valor de asociacion X es la beta
print(model.params)

# El valor de la recta es: Y = 7.03 + 0.04 * x
# Imprimimos la informacion asociada
print(model.summary())

# Podemos predecir los valores de cada X con el modelo mediante el metodo predict, el cual le pasamos como parametros un array de numpy
sales_pred = model.predict( data['TV'] )
print(sales_pred)

# Observamos si los valores reales se ajustan a la predicicon
plt.plot( data['TV'], sales_pred, lw=2, c='g', label = "Recta de regresión")
plt.plot( data['TV'], data['Sales'], 'ro', label="Valores reales")
plt.legend(loc = "best")
plt.xlabel('Sales of TVs')
plt.ylabel('TVs')
plt.title('Linear Regression')
plt.show()

# Obtenemos el valor de prediccion de acuerdo a cada X y la agregamos al dataframe
data['sales_pred'] = 7.032594 + 0.047537 * data['TV']

# Obtenemos el numerador
data['RSE'] = ( data['Sales'] - data['sales_pred'] )**2

# Obtenemos el SSD
SSD = sum(data['RSE'])

# Obtenemos el RSE
RSE = np.sqrt( SSD / (len(data) - 2 ) )

#Obtenemos el promedio
sales_m = np.mean(data['Sales'])

# Obtenemos el error
error = RSE / sales_m
print(f'Error: {error}')
print(data.head())
