import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from plotly.subplots import make_subplots

def gradient(w, X, y):
    return 2 * (X.T @ (X @ w) - X.T @ y) / len(X)

def stochastic_gradient_descent(x, y, w, alpha, num_iterations=300, print_progress=100, seed=None):

    print(f"Iteration 0. Intercept {w[0]:.2f}. Slope {w[1]:.2f}.")
    iterations = 1  # init iterations
    if seed is not None:  # init seed (if given)
        np.random.seed(seed)

    while iterations <= num_iterations:
        i = np.random.randint(len(x))  # <--- this is the only new bit! <---
        g = gradient(w, x[i, None], y[i, None])  # calculate current gradient
        w -= alpha * g  # adjust w based on gradient * learning rate
        if iterations % print_progress == 0:  # periodically print progress
            print(f"Iteration {iterations}. Intercept {w[0]:.2f}. Slope {w[1]:.2f}.")
        iterations += 1  # increase iteration

    print("Terminated!")
    print(f"Iteration {iterations - 1}. Intercept {w[0]:.2f}. Slope {w[1]:.2f}.")


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

def plot_gradient(x, y, w):
    """MSE gradient."""
    y_hat = x @ w
    error = y - y_hat
    gradient = -(1.0 / len(x)) * 2 * x.T @ error
    mse = (error ** 2).mean()
    return gradient, mse

def plot_stochastic_gradient_descent(
    x,
    y,
    w,
    alpha,
    tolerance: float = 2e-5,
    max_iterations: int = 1000,
    verbose: bool = False,
    print_progress: int = 10,
    history: bool = False,
    seed=None,
):
    """MSE stochastic gradient descent."""
    if seed is not None:
        np.random.seed(seed)
    iterations = 1
    if verbose:
        print(f"Iteration 0.", "Weights:", [f"{_:.2f}" for _ in w])
    if history:
        ws = []
        mses = []
    while True:
        i = np.random.randint(len(x))
        g, mse = plot_gradient(x[i, None], y[i, None], w)
        if history:
            ws.append(list(w))
            mses.append(mse)
        w_new = w - alpha * g
        if sum(abs(w_new - w)) < tolerance:
            if verbose:
                print(f"Converged after {iterations} iterations!")
                print("Final weights:", [f"{_:.2f}" for _ in w_new])
            break
        if iterations % print_progress == 0:
            if verbose:
                print(
                    f"Iteration {iterations}.",
                    "Weights:",
                    [f"{_:.2f}" for _ in w_new],
                )
        iterations += 1
        if iterations > max_iterations:
            if verbose:
                print(f"Reached max iterations ({max_iterations})!")
                print("Final weights:", [f"{_:.2f}" for _ in w_new])
            break
        w = w_new
    if history:
        w = w_new
        _, mse = plot_gradient(x, y, w)
        ws.append(list(w))
        mses.append(mse)
        return ws, mses

def plot_gradient_descent_2d(
    x,
    y,
    w,
    alpha,
    m_range,
    b_range,
    tolerance=2e-5,
    max_iterations=5000,
    step_size=1,
    markers=False,
    seed=None,
):
    if x.ndim == 1:
        x = np.array(x).reshape(-1, 1)
        weights, losses = plot_stochastic_gradient_descent(
            np.hstack((np.ones((len(x), 1)), x)),
            y,
            w,
            alpha,
            tolerance,
            max_iterations,
            history=True,
            seed=seed,
        )
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