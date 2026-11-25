"""
blendfit.py
-----------
The model-blend fit, shared by the live bot (munich_bot.get_daily_blend) and
the historical backtest, so both use exactly the same method.

Finds model weights (each 0-1, summing to 1) and a bias (-2..+2 °C) that make
    blend = sum(weight_i * forecast_i) + bias
miss the station value by as little as possible (mean squared error).
"""

import numpy as np
from scipy.optimize import minimize


def fit_blend(X_tr, y_tr):
    """X_tr: (days x models) forecasts, y_tr: (days,) station values.
    Returns (weights array, bias)."""
    num_models = X_tr.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w[:num_models]) - 1.0})
    bounds = [(0.0, 1.0)] * num_models + [(-2.0, 2.0)]
    initial_guess = [1.0 / num_models] * num_models + [0.0]

    def loss_func(params):
        w, b = params[:num_models], params[num_models]
        return np.mean((y_tr - (np.dot(X_tr, w) + b)) ** 2)

    res = minimize(loss_func, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x[:num_models], float(res.x[num_models])
