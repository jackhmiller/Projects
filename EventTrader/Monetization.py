import numpy as np


def get_real_returns(data, event_date, day_range, direction):
    start = data.index.get_loc(event_date.strftime("%Y-%m-%d")) + day_range[0]
    end = data.index.get_loc(event_date.strftime("%Y-%m-%d")) + day_range[1]
    leverage = var_adj(data, start)
    y_true = data.iloc[start: end]
    real_returns = y_true*leverage
    y_true = int(np.sign(sum(real_returns) * direction))

    return real_returns, y_true


def var_adj(data, start):
    """ 2-year rolling VaR, with avg of past three months used as final"""
    return abs((0.015 / data.iloc[:start].rolling(500).quantile(0.05))[-90:].mean())