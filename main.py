import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import *


'''prendiamo il dataset'''
df = (pd.read_csv("data/pokemon.csv", usecols=['name', 'defense', 'attack'], index_col=0)
        .reset_index()
     )
x = StandardScaler().fit_transform(df[['defense']]).flatten()
y = df['attack'].to_numpy()
#plot_pokemon(x, y, x_range=[0, 1], y_range=[0, 200], dx=0.2, dy=50)


'''aggiungiamo una colonna di "1" per rappresentare l'intercetta'''
X = np.hstack((np.ones((len(x), 1)), x[:, None]))
#print(X)


'''calcoliamo il gradiente stocastico'''
w = [-20, -5]
alpha = 0.01
X = np.hstack((np.ones((len(x), 1)), x[:, None]))
weights = stochastic_gradient_descent(X, y, w, alpha, seed=2020)

'''visualizziamo la path'''
plot_gradient_descent_2d(x, y, weights, np.arange(-30, 60, 2), np.arange(-40, 140, 2))