import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import *


'''
Prendiamo il dataset
- pd.read_csv: Carica il file CSV con i dati dei Pokémon
    * usecols: Seleziona solo le colonne name, defense, e attack.
'''
df = pd.read_csv("data/pokemon.csv", usecols=['name', 'defense', 'attack'])

'''
Normalizza i valori della colonna defense, scalando la media a 0 e la deviazione standard a 1 eliminando il problema
delle diverse scale, migliorare l'efficienza degli algoritmi e rendere i dati più comprensibili.
- .flatten(): Converte l’output in un array monodimensionale.
Risultato: x contiene i valori normalizzati della colonna defense.
'''
x = StandardScaler().fit_transform(df[['defense']]).flatten()

'''
Converte i valori della colonna attack in un array NumPy.
Risultato: y contiene i valori della colonna attack in forma numerica.
'''
y = df['attack'].to_numpy()

'''
Stampa pokemon
'''
plot_pokemon(x, y, x_range=[0, 1], y_range=[0, 200], dx=0.2, dy=50)


'''
Aggiungiamo una colonna di "1" per rappresentare l'intercetta, per permettere al modello di adattarsi ai dati
- np.ones((len(x), 1)): Crea una colonna di 1 con lunghezza pari a quella di x (bias term per la regressione lineare).
- x[:, None]: Converte l'array x (monodimensionale) in una matrice colonna
- np.hstack: Concatena la colonna di 1 con la matrice colonna x
Risultato: X è una matrice con due colonne: una di 1 e una con i valori normalizzati di x.'''
X = np.hstack((np.ones((len(x), 1)), x[:, None]))
#print(X)

'''
Inizializziamo vettore dei pesi iniziali con valori arbirtari (lontani dall'ottimo per analizzare il processo di convergenza)
- w[0] intercetta
- w[1] pendenza
'''
w = [-20, -5]

'''
Inizializzamo il tasso di apprendimento per l’algoritmo di discesa del gradiente.
Un valore piccolo (come 0.01) consente aggiornamenti graduali, riducendo il rischio di saltare l’ottimo.
'''
alpha = 0.01

'''
Calcolo gradiente
- stochastic_gradient_descent: implementa l'algoritmo della discesa del gradiente stocastica per ottimizzare i pesi w.
    * X: Matrice delle feature (con la colonna di bias e i valori normalizzati di x).
    * y: Array dei valori target (valori della colonna attack).
    * w: Vettore dei pesi iniziali (bias e pendenza).
    * alpha: Tasso di apprendimento
    * seed: Seme per generare numeri casuali.
Risultato: Restituisce i pesi aggiornati `weights` che minimizzano l'errore.
'''
weights = stochastic_gradient_descent(X, y, w, alpha, num_iterations=300, seed=2020)

'''Visualizziamo la path di discesa del grandiente
- plot_gradient_descent_2d: funzione di visualizzazione del percorso in uno spazio bidimensionale
    * x, y: Dati normalizzati (x: feature, y: target).
    * weights: Pesi aggiornati dopo l'ottimizzazione.
    * np.arange(-30, 60, 2): Range di valori per la visualizzazione dell'asse del bias w[0].
    * np.arange(-40, 140, 2): Range di valori per la visualizzazione dell'asse del coefficiente w[1].
Risultato: Un grafico che mostra come i pesi `weights` si avvicinano all'ottimo durante le iterazioni.
'''
plot_gradient_descent_2d(x, y, weights, np.arange(-30, 60, 2), np.arange(-40, 140, 2))