import sys
import datetime as dt
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QIcon, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QDateEdit, QTabWidget, QProgressBar,
    QMessageBox, QTableWidget, QTableWidgetItem, QSizePolicy, QHeaderView
)

import pyqtgraph as pg
from pyqtgraph import DateAxisItem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import lightgbm as lgb
import shap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

CACHE_DIR = Path.home() / ".stock_recommender_cache"
CACHE_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET_HORIZON_DAYS = 21  # прогноз на 21 день


def log(message: str):
    print(dt.datetime.now().strftime("%H:%M:%S"), message)


class DataLoader(QThread):
    finished = Signal(pd.DataFrame)
    progress = Signal(int)
    error = Signal(str)

    def __init__(self, tickers: List[str], start_date: dt.date, end_date: dt.date):
        super().__init__()
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        try:
            df = self._download()
            self.finished.emit(df)
        except Exception as exc:
            self.error.emit(str(exc))

    def _download(self) -> pd.DataFrame:
        self.progress.emit(5)
        # Добавляем future_stack=True, чтобы не получать предупреждение
        raw = yf.download(
            tickers=" ".join(self.tickers),
            start=self.start_date,
            end=self.end_date + dt.timedelta(days=1),
            progress=False,
            auto_adjust=True,
            threads=True,
        )
        # Используем future_stack=True для будущей версии pandas
        df = (
            raw.stack(level=1, future_stack=True)
            .rename_axis(index={"Date": "date", "Ticker": "ticker"})
            .reset_index()
        )
        self.progress.emit(30)
        if df.empty:
            raise ValueError("Нет данных. Проверьте тикеры или диапазон дат.")
        return df


class FeatureBuilder:
    def __init__(self, horizon: int = TARGET_HORIZON_DAYS):
        self.horizon = horizon

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["return"] = df.groupby("ticker")["Close"].pct_change(fill_method=None)
        for win in (5, 10, 21, 63, 126, 252):
            df[f"ret_{win}"] = df.groupby("ticker")["Close"].pct_change(win, fill_method=None)
            df[f"sma_{win}"] = df.groupby("ticker")["Close"].transform(lambda s: s.rolling(win).mean())
            df[f"vol_{win}"] = df.groupby("ticker")["return"].transform(lambda s: s.rolling(win).std())
        df["target"] = (
            df.groupby("ticker")["Close"]
            .shift(-self.horizon)
            .pct_change(self.horizon, fill_method=None)
        )
        df = df.dropna()
        return df


class ModelTrainer(QThread):
    progress = Signal(int)
    finished = Signal(object, pd.DataFrame)
    error = Signal(str)

    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data

    def run(self):
        try:
            model, feat_df = self._train()
            self.finished.emit(model, feat_df)
        except Exception as exc:
            self.error.emit(str(exc))

    def _train(self):
        self.progress.emit(10)
        df = self.data.copy()
        y = df.pop("target")
        X = df.select_dtypes(include=[np.number])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=False
        )
        self.progress.emit(40)

        model = lgb.LGBMRegressor(
            n_estimators=400,
            max_depth=-1,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
        )
        self.progress.emit(80)

        preds = model.predict(X_val)
        r2 = r2_score(y_val, preds)
        log(f"Validation R²: {r2:.3f}")

        self.progress.emit(90)
        shap_vals = shap.TreeExplainer(model).shap_values(X_val[:200])
        feat_df = (
            pd.DataFrame({"feature": X.columns, "importance": np.abs(shap_vals).mean(axis=0)})
            .sort_values("importance", ascending=False)
            .head(25)
        )
        self.progress.emit(100)
        return model, feat_df


class Recommender:
    def __init__(self, model, builder: FeatureBuilder):
        self.model = model
        self.builder = builder

    def recommend(self, raw_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        df = self.builder.transform(raw_df)
        X = df.select_dtypes(include=[np.number]).drop(columns=["target"])
        df["pred"] = self.model.predict(X)

        latest = df.sort_values("date").groupby("ticker").tail(1)
        latest["score"] = latest["pred"] / latest["vol_21"].clip(lower=1e-6)
        latest["last_close"] = latest["Close"]
        latest["volatility_21"] = latest["vol_21"]

        def risk_level(vol):
            if vol < 0.02:
                return "Низкий"
            elif vol < 0.05:
                return "Средний"
            else:
                return "Высокий"

        latest["risk_level"] = latest["volatility_21"].apply(risk_level)

        scores = latest["score"].clip(lower=0)
        total = scores.sum() if scores.sum() > 0 else 1
        latest["allocation_pct"] = (scores / total * 100).round(2)

        ranked = latest.sort_values("score", ascending=False).head(top_n)[
            ["ticker", "last_close", "pred", "volatility_21", "risk_level", "allocation_pct", "score"]
        ].copy()

        ranked.rename(columns={
            "ticker": "Тикер",
            "last_close": "Последняя цена",
            "pred": "Прогноз доходности",
            "volatility_21": "Волатильность (21 дн)",
            "risk_level": "Уровень риска",
            "allocation_pct": "Распределение (%)",
            "score": "Скоринг"
        }, inplace=True)

        ranked["Прогноз доходности"] = (ranked["Прогноз доходности"] * 100).round(2).astype(str) + " %"
        ranked["Волатильность (21 дн)"] = (ranked["Волатильность (21 дн)"] * 100).round(2).astype(str) + " %"

        # Оставляем 'Распределение (%)' как float, чтобы строить диаграмму без ошибок
        return ranked


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Рекомендатор Акций 2025 — ML")
        self.setWindowIcon(QIcon.fromTheme("chart-line"))
        self.setMinimumSize(1200, 800)

        # Устанавливаем темную тему Fusion
        QApplication.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
        palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.Highlight, QColor(142, 45, 197))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        QApplication.setPalette(palette)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Панель управления
        ctrl_layout = QHBoxLayout()
        layout.addLayout(ctrl_layout)

        ctrl_layout.addWidget(QLabel("Тикеры (через запятую):"))
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Например: AAPL, MSFT, AMZN")
        self.ticker_input.setMinimumWidth(300)
        ctrl_layout.addWidget(self.ticker_input)

        self.examples = QLabel(
            "Примеры тикеров: AAPL (Apple), MSFT (Microsoft), AMZN (Amazon),\n"
            "GOOGL (Alphabet), TSLA (Tesla), NVDA (NVIDIA), SPY (ETF S&P 500), JPM (JPMorgan)"
        )
        self.examples.setWordWrap(True)
        layout.addWidget(self.examples)

        ctrl_layout.addWidget(QLabel("Начало:"))
        self.start_date = QDateEdit()
        self.start_date.setDate(dt.date.today() - dt.timedelta(days=365 * 5))
        self.start_date.setCalendarPopup(True)
        ctrl_layout.addWidget(self.start_date)

        ctrl_layout.addWidget(QLabel("Конец:"))
        self.end_date = QDateEdit()
        self.end_date.setDate(dt.date.today())
        self.end_date.setCalendarPopup(True)
        ctrl_layout.addWidget(self.end_date)

        self.fetch_btn = QPushButton("Загрузить данные")
        ctrl_layout.addWidget(self.fetch_btn)

        self.train_btn = QPushButton("Обучить модель")
        self.train_btn.setEnabled(False)
        ctrl_layout.addWidget(self.train_btn)

        self.rec_btn = QPushButton("Показать рекомендации")
        self.rec_btn.setEnabled(False)
        ctrl_layout.addWidget(self.rec_btn)

        ctrl_layout.addStretch()

        # Прогресс-бар
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # Вкладки
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.tab_data = QWidget()
        self.tab_chart = QWidget()
        self.tab_features = QWidget()
        self.tab_recommend = QWidget()

        self.tabs.addTab(self.tab_data, "Данные")
        self.tabs.addTab(self.tab_chart, "Графики")
        self.tabs.addTab(self.tab_features, "Важность признаков")
        self.tabs.addTab(self.tab_recommend, "Рекомендации")

        # Вкладка «Данные»
        self.table_data = QTableWidget()
        self.table_data.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tab_data.setLayout(QVBoxLayout())
        self.tab_data.layout().addWidget(self.table_data)

        # Вкладка «Графики» – используем DateAxisItem для корректного отображения дат
        self.chart = pg.PlotWidget(
            axisItems={'bottom': DateAxisItem(orientation='bottom')},
            background="#111111"
        )
        self.chart.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
        self.chart.getPlotItem().setLabel("bottom", "Дата", color="#CCCCCC")
        self.chart.getPlotItem().setLabel("left", "Цена (USD)", color="#CCCCCC")
        self.tab_chart.setLayout(QVBoxLayout())
        self.tab_chart.layout().addWidget(self.chart)

        # Вкладка «Важность признаков»
        self.table_features = QTableWidget()
        self.tab_features.setLayout(QVBoxLayout())
        self.tab_features.layout().addWidget(self.table_features)

        # Вкладка «Рекомендации»
        self.table_recommend = QTableWidget()
        self.pie_canvas = FigureCanvas(plt.Figure(figsize=(4, 4), dpi=100))
        self.pie_canvas.figure.patch.set_facecolor("#111111")
        self.pie_ax = self.pie_canvas.figure.add_subplot(111)
        self.tab_recommend.setLayout(QHBoxLayout())
        self.tab_recommend.layout().addWidget(self.table_recommend, 3)
        self.tab_recommend.layout().addWidget(self.pie_canvas, 2)

        # Подключаем кнопки
        self.fetch_btn.clicked.connect(self.handle_fetch)
        self.train_btn.clicked.connect(self.handle_train)
        self.rec_btn.clicked.connect(self.handle_recommend)

        self.raw_df: pd.DataFrame | None = None
        self.model = None
        self.builder = FeatureBuilder()

    def handle_fetch(self):
        tickers = [t.strip().upper() for t in self.ticker_input.text().split(",") if t.strip()]
        if not tickers:
            QMessageBox.warning(self, "Ошибка ввода", "Введите хотя бы один тикер.")
            return

        start = self.start_date.date().toPython()
        end = self.end_date.date().toPython()

        self.fetch_btn.setEnabled(False)
        self.progress.setValue(0)

        self.loader = DataLoader(tickers, start, end)
        self.loader.progress.connect(self.progress.setValue)
        self.loader.finished.connect(self.on_data_loaded)
        self.loader.error.connect(self.on_error)
        self.loader.start()

    @Slot(pd.DataFrame)
    def on_data_loaded(self, df: pd.DataFrame):
        self.raw_df = df
        display_df = df.head(1000)
        self.populate_table(self.table_data, display_df)
        self.update_chart(df)
        self.train_btn.setEnabled(True)
        self.fetch_btn.setEnabled(True)
        self.progress.setValue(100)

    @Slot()
    def handle_train(self):
        if self.raw_df is None:
            return
        self.train_btn.setEnabled(False)
        self.progress.setValue(0)

        features_df = self.builder.transform(self.raw_df)
        self.trainer = ModelTrainer(features_df)
        self.trainer.progress.connect(self.progress.setValue)
        self.trainer.finished.connect(self.on_model_trained)
        self.trainer.error.connect(self.on_error)
        self.trainer.start()

    @Slot(object, pd.DataFrame)
    def on_model_trained(self, model, feat_df: pd.DataFrame):
        self.model = model
        self.populate_table(self.table_features, feat_df)
        self.rec_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.progress.setValue(100)

    @Slot()
    def handle_recommend(self):
        if self.model is None or self.raw_df is None:
            return
        recommender = Recommender(self.model, self.builder)
        rec_df = recommender.recommend(self.raw_df, top_n=15)
        self.populate_table(self.table_recommend, rec_df)

        # Строим круговую диаграмму распределения
        self.pie_ax.clear()
        labels = rec_df["Тикер"].tolist()
        sizes = rec_df["Распределение (%)"].tolist()  # Уже float, без strip
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
        wedges, texts, autotexts = self.pie_ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            textprops={"color": "white"},
            wedgeprops={"edgecolor": "#222222"}
        )
        self.pie_ax.set_facecolor("#111111")
        self.pie_ax.set_title("Распределение портфеля", color="white")
        for text in texts + autotexts:
            text.set_color("white")
        self.pie_canvas.draw()

        self.tabs.setCurrentWidget(self.tab_recommend)

    @Slot(str)
    def on_error(self, msg: str):
        QMessageBox.critical(self, "Ошибка", msg)
        self.progress.setValue(0)
        self.fetch_btn.setEnabled(True)
        self.train_btn.setEnabled(self.raw_df is not None)

    def populate_table(self, widget: QTableWidget, df: pd.DataFrame):
        widget.clear()
        widget.setRowCount(len(df))
        widget.setColumnCount(len(df.columns))
        widget.setHorizontalHeaderLabels(df.columns.astype(str).tolist())
        widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for i, (_, row) in enumerate(df.iterrows()):
            for j, val in enumerate(row):
                text = f"{val}" if not isinstance(val, float) else f"{val:.4f}"
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                widget.setItem(i, j, item)

    def update_chart(self, df: pd.DataFrame):
        self.chart.clear()
        for ticker, data in df.groupby("ticker"):
            # Преобразуем даты в секунды с начала эпохи и получаем numpy-массив
            x_vals = (pd.to_datetime(data["date"]).astype(np.int64) // 10**9).to_numpy()
            y_vals = data["Close"].to_numpy()
            pen = pg.mkPen(color=pg.intColor(hash(ticker) % 256, 256), width=2, style=Qt.SolidLine)
            self.chart.plot(
                x=x_vals,
                y=y_vals,
                pen=pen,
                name=ticker
            )
        self.chart.getPlotItem().addLegend(labelTextColor="#FFFFFF")
        self.chart.getPlotItem().setLabel("bottom", "Дата", color="#CCCCCC")
        self.chart.getPlotItem().setLabel("left", "Цена (USD)", color="#CCCCCC")


def main():
    pg.setConfigOptions(antialias=True, background="#111111", foreground="#FFFFFF")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
