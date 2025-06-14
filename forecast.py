import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import random, os

class ForecastModel:
    def __init__(self, look_back=60, horizon=14, epochs=20, seed=42):
        self.look_back = look_back
        self.horizon = horizon
        self.epochs = epochs
        self.seed = seed
        self._fix_seed()

    def _fix_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

    def _load(self, ticker, period="5y"):
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("Нет данных по тикеру")
        return df

    def _prepare(self, series):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.reshape(-1,1))
        X, y = [], []
        for i in range(len(scaled) - self.look_back - self.horizon):
            X.append(scaled[i:i+self.look_back])
            y.append(scaled[i+self.look_back:i+self.look_back+self.horizon].flatten())
        return np.array(X), np.array(y), scaler

    def _build(self):
        # Вход
        inp = Input(shape=(self.look_back, 1))
        # LSTM-слой
        x = LSTM(128, return_sequences=True)(inp)
        x = LayerNormalization()(x)
        # Multi-head attention: (query=x, value=x)
        attn = MultiHeadAttention(num_heads=4, key_dim=16)(query=x, value=x)
        x = LayerNormalization()(attn)
        # Ещё LSTM
        x = LSTM(64)(x)
        out = Dense(self.horizon)(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit_predict(self, ticker):
        df = self._load(ticker)
        prices = df["Close"].values
        X, y, scaler = self._prepare(prices)
        if len(X) < 200:
            raise ValueError("Недостаточно данных для обучения")
        model = self._build()
        es = EarlyStopping(patience=4, restore_best_weights=True, verbose=0)
        model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0, callbacks=[es])
        # Прогноз
        last_window = scaler.transform(prices[-self.look_back:].reshape(-1,1))
        last_window = last_window.reshape(1, self.look_back, 1)
        pred_scaled = model.predict(last_window)[0]
        preds = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
        return preds, df
