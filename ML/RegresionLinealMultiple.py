import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Leemos el csv
data = pd.read_csv('./datasets/ads/Advertising.csv')

# Establecemos la regresion multiple como una relacion entre las ventas y los productos
# asociados a TV y Newspaper

model = smf.ols('Sales~TV+Newspaper', data=data).fit()

# Imprimimos los variables de la ecuacion
print(model.params)

# Imprimiemos la informacion general
print(model.summary())

# Hacemos unas predicciones, como tenemos dos variables entonces hay que pasarle como argumentos dos columnas
sales_pred = model.predict( data[['TV', 'Newspaper']] )

# SSD
SSD = sum( (data['Sales'] - sales_pred)** 2 )

# RSE
RSE =  np.sqrt( SSD/( len(data) - 2 - 1 ) )

# Calculamos el error
sales_m = np.mean( data['Sales'] )
error = RSE / sales_m

# MULTICOLINEALIDAD
# De acuerdo con el VIF (Factor de Inflacion de la Varianza) si
# VIF = 1 Las variables no estan correlacionadas y pueden permanecer en el modelo
# VIF < 5 Las variables variables tienen una correlacion moderada y se pueden quedar en el modelo
# VIF > 5 Las variables estas altamente correlacionadas y hay que eliminarlas del modelo

# NewsPaper ~ TV + Radio -> R^2 VIF = 1/(1-R^2)
lm_n = smf.ols('Newspaper~TV+Radio', data = data).fit()
rsquared_n = lm_n.rsquared
VIF = 1/ ( 1- rsquared_n )
print(VIF)

# TV ~ Radio + NewsPaper -> R^2 VIF = 1/(1-R^2)
lm_t = smf.ols('TV~Newspaper+Radio', data = data).fit()
rsquared_t = lm_t.rsquared
VIF = 1/ ( 1- rsquared_t )
print(VIF)

# Radio ~ TV + NewsPaper -> R^2 VIF = 1/(1-R^2)
lm_r = smf.ols('Radio~TV+Newspaper', data = data).fit()
rsquared_r = lm_r.rsquared
VIF = 1/ ( 1- rsquared_r )
print(VIF)

# En los resultados se ve que Newspaper y Radio tienen un VIF similar por lo cual se puede optar por eliminar
# alguna de las dos ya que estan medianamente correlacionadas, el que mejor aporta al modelo es la variable
# radio







