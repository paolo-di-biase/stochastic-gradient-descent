import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import *


'''prendiamo il dataset'''
# pd.read_csv: Carica il file CSV con i dati dei Pokémon
#   usecols: Seleziona solo le colonne name, defense, e attack.
#   index_col=0: Usa la prima colonna (name) come indice.
# .reset_index(): Ripristina l’indice predefinito (aggiungendo il nome come colonna).
df = (pd.read_csv("data/pokemon.csv", usecols=['name', 'defense', 'attack'], index_col=0)
        .reset_index()
     )
# StandardScaler().fit_transform: Normalizza i valori della colonna defense, scalando la media a 0 e la deviazione standard a 1.
#   .flatten(): Converte l’output in un array monodimensionale.
# Risultato: x contiene i valori normalizzati della colonna defense.
x = StandardScaler().fit_transform(df[['defense']]).flatten()
# .to_numpy(): Converte i valori della colonna attack in un array NumPy.
# Risultato: y contiene i valori della colonna attack in forma numerica.3
y = df['attack'].to_numpy()
#plot_pokemon(x, y, x_range=[0, 1], y_range=[0, 200], dx=0.2, dy=50)

# Creazione della matrice delle feature
'''aggiungiamo una colonna di "1" per rappresentare l'intercetta'''
# np.ones((len(x), 1)): Crea una colonna di 1 con lunghezza pari a quella di x (bias term per la regressione lineare).
#   La colonna di 1 viene aggiunta per includere il termine di bias (o intercetta) nella regressione lineare.
#   Cos’è il bias?
# 	    •	Il bias consente al modello di adattarsi ai dati quando il valore di tutte le feature (in questo caso x) è 0.
#   Senza la colonna di 1:
# 	    •	Il modello non avrebbe un termine di bias.
# 	    •	Sarebbe costretto a passare attraverso l’origine (y = 0 quando x = 0), limitando la sua capacità di adattarsi ai dati.
# x[:, None]: Converte x in una matrice colonna, cioè trasforma l’array x (monodimensionale) in una matrice colonna
#       (dimensione: numero di righe di x x 1). Questo è necessario perché x deve essere concatenato come una matrice
# np.hstack: Combina la colonna di 1 con x lungo l’asse orizzontale.
# Risultato: X è una matrice con due colonne: una di 1 e una con i valori normalizzati di x.
X = np.hstack((np.ones((len(x), 1)), x[:, None]))
#print(X)


'''calcoliamo il gradiente stocastico'''
# w: Vettore di pesi iniziali per il modello di regressione lineare (bias e coefficiente della feature).
# 	Significato di w:
# 	•	w[0] è il bias term (intercetta).
# 	•	w[1] è il coefficiente per la feature x (pendenza).
# 	Perché -20 e -5?
# 	•	Valori arbitrari per inizializzare i pesi. L’algoritmo di discesa del gradiente correggerà questi valori per minimizzare l’errore tra il modello predetto e i valori effettivi di y.
# 	•	Valori iniziali lontani dall’ottimo possono servire per analizzare il processo di convergenza.
w = [-20, -5]
# alpha: Tasso di apprendimento per l’algoritmo di discesa del gradiente.
# 	Significato di alpha:
# 	    È il tasso di apprendimento, che controlla quanto velocemente i pesi w vengono aggiornati ad ogni iterazione della discesa del gradiente.
# 	Perché 0.01?
# 	•	Un valore piccolo (come 0.01) consente aggiornamenti graduali, riducendo il rischio di saltare l’ottimo.
# 	•	Valori più grandi potrebbero causare instabilità o divergenza, mentre valori troppo piccoli rallenterebbero la convergenza.
alpha = 0.01
# stochastic_gradient_descent:
#   Questa funzione implementa l'algoritmo della discesa del gradiente stocastica per ottimizzare i pesi w.
#   Parametri:
#       - X: Matrice delle feature (con la colonna di bias e i valori normalizzati di x).
#       - y: Array dei valori target (valori della colonna attack).
#       - w: Vettore dei pesi iniziali (bias e coefficiente per la feature).
#       - alpha: Tasso di apprendimento che determina la dimensione del passo verso l'ottimizzazione.
#       - seed: Seme per generare numeri casuali (riproducibilità nei risultati).
#   Funzionamento:
#       - Aggiorna iterativamente i pesi `w` calcolando il gradiente della funzione di perdita per un singolo campione scelto casualmente.
#       - Obiettivo: minimizzare l'errore (ad esempio, l'errore quadratico medio) tra i valori predetti e quelli reali.
#   Risultato:
#       - Restituisce i pesi aggiornati `weights` che minimizzano l'errore.
weights = stochastic_gradient_descent(X, y, w, alpha, seed=2020)

'''visualizziamo la path'''
# plot_gradient_descent_2d:
#   Questa funzione visualizza il percorso della discesa del gradiente in uno spazio bidimensionale.
#   Parametri:
#       - x, y: Dati normalizzati (x: feature, y: target).
#       - weights: Pesi aggiornati dopo l'ottimizzazione.
#       - np.arange(-30, 60, 2): Range di valori per la visualizzazione dell'asse del bias w[0].
#       - np.arange(-40, 140, 2): Range di valori per la visualizzazione dell'asse del coefficiente w[1].
#   Funzionamento:
#       - Rappresenta un piano 2D dei pesi w[0] (bias) e w[1] (coefficiente).
#       - Traccia il percorso della discesa del gradiente nel processo di minimizzazione dell'errore.
#   Risultato:
#       - Un grafico che mostra come i pesi `weights` si avvicinano all'ottimo durante le iterazioni.
plot_gradient_descent_2d(x, y, weights, np.arange(-30, 60, 2), np.arange(-40, 140, 2))