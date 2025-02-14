import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from plotly.subplots import make_subplots

'''
Funzione di calcolo gradiente della funzione di perdita dei minimi quadrati rispetto ai pesi del modello
Input:
- w, array dei pesi del modello
- X, matrice degli input
- y, array dei valori target

La funzione calcola il gradiente del MSE rispetto ai pesi w secondo la formula:
∇MSE(w)=(2/n)(X.T(y-y'))
ottimizzata in modo da effettuare il prodotto matrice-vettore che è più efficiente
'''
def gradient(w, X, y):
    y_pred = (X @ w)
    return 2 * (X.T @ y_pred - X.T @ y) / len(X)

'''
Funzione di calcolo della discesa stocastica del gradiente, per ottimizzare i pesi
Input:
- x, matrice degli input
- y, array dei valori target
- w, vettore dei pesi inizali
- alpha, learning rate
- num_iterations, numero massimo di iterazioni da eseguire
- print_progress, frequenza con cui stampare i progressi
- seed, seed per la pseudorandomness

La funzione aggiorna i pesi in modo da ottimizzare l'errore quadratico medio, sfrutta quindi il gradiente, ma 
l'aggiornamento è fatto usando un singolo esempio casuale per ogni iterazione.
'''
def stochastic_gradient_descent(x, y, w, alpha, num_iterations=300, print_progress=100, seed=None):

    ws=[] # vettore che contiene i vari aggiornamenti dei pesi, usato per il plot

    print(f"Iteration 0. Intercept {w[0]:.2f}. Slope {w[1]:.2f}.")
    iterations = 1
    if seed is not None:
        np.random.seed(seed)

    epsilon = 1e-4  # inizializzato epsilon per criterio di stop
    g = np.inf  # inizializzato g a valore grande
    while np.linalg.norm(g) >= epsilon and iterations < num_iterations:  # criteri di stop: gradiente < epsilon || raggiunto num max iterazioni
        i = np.random.randint(len(x))  # selezione di un indice casuale
        g = gradient(w, x[i, None], y[i, None])  # calcolo del gradiente sull'elemento selezionato
        w -= alpha * g  # aggiustati pesi tramite il gradiente * learning rate
        ws.append(list(w)) # aggiunti pesi al vettore
        if iterations % print_progress == 0:  # stampa progressi se necessario
            print(f"Iteration {iterations}. Intercept {w[0]:.2f}. Slope {w[1]:.2f}.")
        iterations += 1  # incrementa iterazioni

    print("Terminated!")
    print(f"Iteration {iterations - 1}. Intercept {w[0]:.2f}. Slope {w[1]:.2f}.")
    return ws


'''**********************************************PLOT FUNCTIONS**************************************************'''

def plot_pokemon(
    x, y, y_hat=None, x_range=[10, 130], y_range=[10, 130], dx=20, dy=20
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", marker=dict(size=10), name="data")
    )
    if y_hat is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_hat,
                line_color="red",
                mode="lines",
                line=dict(width=3),
                name="Fitted line",
            )
        )
        width = 550
        title_x = 0.46
    else:
        width = 500
        title_x = 0.5
    fig.update_layout(
        width=width,
        height=500,
        title="Pokemon stats",
        title_x=title_x,
        title_y=0.93,
        xaxis_title="defense",
        yaxis_title="attack",
        margin=dict(t=60),
    )
    fig.update_xaxes(range=x_range, tick0=x_range[0], dtick=dx)
    fig.update_yaxes(range=y_range, tick0=y_range[0], dtick=dy)
    fig.show()
    return fig

def plot_gradient_descent_2d(
    x,
    y,
    weights,
    m_range,
    b_range,
    step_size=1,
    markers=False,
):
    if x.ndim == 1:
        x = np.array(x).reshape(-1, 1)
        title = "Stochastic Gradient Descent"
    weights = np.array(weights)
    intercepts, slopes = weights[:, 0], weights[:, 1]
    mse = np.zeros((len(m_range), len(b_range)))
    for i, slope in enumerate(m_range):
        for j, intercept in enumerate(b_range):
            mse[i, j] = mean_squared_error(y, x * slope + intercept)

    fig = make_subplots(
        rows=1,
        subplot_titles=[title],  # . Iterations = {len(intercepts) - 1}."],
    )
    fig.add_trace(
        go.Contour(z=mse, x=b_range, y=m_range, name="", colorscale="viridis")
    )
    mode = "markers+lines" if markers else "lines"
    fig.add_trace(
        go.Scatter(
            x=intercepts[::step_size],
            y=slopes[::step_size],
            mode=mode,
            line=dict(width=2.5),
            line_color="coral",
            marker=dict(
                opacity=1,
                size=np.linspace(19, 1, len(intercepts[::step_size])),
                line=dict(width=2, color="DarkSlateGrey"),
            ),
            name="Descent Path",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[intercepts[0]],
            y=[slopes[0]],
            mode="markers",
            marker=dict(size=20, line=dict(width=2, color="DarkSlateGrey")),
            marker_color="orangered",
            name="Start",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[intercepts[-1]],
            y=[slopes[-1]],
            mode="markers",
            marker=dict(size=20, line=dict(width=2, color="DarkSlateGrey")),
            marker_color="yellowgreen",
            name="End",
        )
    )
    fig.update_layout(
        width=700,
        height=600,
        margin=dict(t=60),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_xaxes(
        title="intercept (w<sub>0</sub>)",
        range=[b_range.min(), b_range.max()],
        tick0=b_range.min(),
        row=1,
        col=1,
        title_standoff=0,
    )
    fig.update_yaxes(
        title="slope (w<sub>1</sub>)",
        range=[m_range.min(), m_range.max()],
        tick0=m_range.min(),
        row=1,
        col=1,
        title_standoff=0,
    )
    fig.show()
    return fig