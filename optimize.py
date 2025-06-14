# optimize.py

import pandas as pd
import yfinance as yf
import numpy as np
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

def suggest_portfolio(tickers):
    raw = yf.download(
        tickers,
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    # Попытаемся взять столбец Close
    try:
        prices = raw["Close"]
    except KeyError:
        prices = raw.xs("Close", axis=1, level=1)

    # Если вернулся Series (один тикер) — превращаем в DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    # Оставляем только столбцы с данными
    prices = prices.dropna(axis=1, how="all")

    # Проверяем, достаточно ли инструментов и точек во времени
    # Нужно минимум 2 столбца и >=2 наблюдений
    if prices.shape[1] < 2 or prices.shape[0] < 2:
        # fallback: равномерное распределение по доступным тикерам
        cols = list(prices.columns)
        if not cols:
            raise ValueError("Нет данных ни для одного тикера — проверьте ввод.")
        w = {t: 1.0 / len(cols) for t in cols}
        # возвращаем NaN для метрик, чтобы UI понял, что расчёт условен
        return w, np.nan, np.nan, np.nan

    # Собственно расчёт оптимального портфеля
    mu = mean_historical_return(prices)
    S = CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned = ef.clean_weights()
    exp_ret, risk, sharpe = ef.portfolio_performance(verbose=False)
    return cleaned, exp_ret, risk, sharpe
