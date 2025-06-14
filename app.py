import streamlit as st
import altair as alt
import yfinance as yf
import pandas as pd
import numpy as np

from forecast import ForecastModel
from optimize import suggest_portfolio

st.set_page_config(page_title="Investment AI Advisor", layout="wide")
st.title("📊 Altair-Powered Investment AI Advisor")
st.markdown("""
**Transformer+LSTM прогнозы**, **технический анализ** и **портфельная оптимизация**  
_Учебный пример, не является финансовым советом._
""")

# ─── Sidebar ────────────────────────────────────────────────
st.sidebar.header("⚙️ Параметры")
tickers_txt = st.sidebar.text_input("Тикеры (через запятую)", "AAPL, MSFT, TSLA, EURUSD=X")
tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
look_back = st.sidebar.slider("Окно обучения (дней)", 30, 240, 60, step=10)
horizon = st.sidebar.slider("Горизонт прогноза (дней)", 1, 60, 14)
epochs = st.sidebar.slider("Эпох обучения", 1, 100, 20)
seed = st.sidebar.number_input("Seed", 42, step=1)
run = st.sidebar.button("🚀 Запустить")

@st.cache_data
def load_history(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )
    if df.empty:
        raise ValueError(f"Нет данных для {ticker}")
    return df

@st.cache_data
def load_ohlc(ticker: str) -> pd.DataFrame:
    df = load_history(ticker)
    # выбираем нужные колонки
    df = df.loc[:, ["Open", "High", "Low", "Close"]]
    if df.isna().all().all():
        raise ValueError(f"Нет OHLC-данных для {ticker}")
    return df

if run and tickers:
    fm = ForecastModel(look_back, horizon, epochs, seed)
    tab1, tab2, tab3 = st.tabs(["📈 Прогнозы", "🛠️ Индикаторы", "💼 Портфель"])

    # ─── Tab 1: Прогнозы ─────────────────────────────────
    with tab1:
        best_tkr, best_delta = None, -np.inf
        for tkr in tickers:
            st.subheader(f"{tkr}: прогноз на {horizon} дней")
            try:
                hist = load_history(tkr)
                # берем последние 3 месяца
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
                hist3m = hist.loc[hist.index >= cutoff]

                preds, _ = fm.fit_predict(tkr)
                # для графика по прогнозу точек horizon
                last_date = hist.index[-1]
                future = pd.date_range(last_date + pd.Timedelta(1, "D"), periods=horizon)

                # готовим данные
                df_hist = hist3m[["Close"]].reset_index().rename(columns={"index": "Date"})
                df_pred = pd.DataFrame({"Date": future, "Forecast": preds})

                # линейный график
                line_hist = (
                    alt.Chart(df_hist)
                    .mark_line(color="steelblue")
                    .encode(
                        x=alt.X("Date:T", title="Дата"),
                        y=alt.Y("Close:Q", title="Цена"),
                        tooltip=[alt.Tooltip("Date:T", title="Дата"),
                                 alt.Tooltip("Close:Q", title="Цена")]
                    )
                )
                line_pred = (
                    alt.Chart(df_pred)
                    .mark_line(color="orange", strokeDash=[5, 5])
                    .encode(
                        x="Date:T",
                        y="Forecast:Q",
                        tooltip=[alt.Tooltip("Date:T", title="Дата"),
                                 alt.Tooltip("Forecast:Q", title="Прогноз")]
                    )
                )

                chart = (
                    (line_hist + line_pred)
                    .properties(height=300)
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)

                # метрика Δ%
                cur, fut = float(hist["Close"].iloc[-1]), float(preds[-1])
                delta = (fut - cur) / cur * 100
                st.metric(label=f"{tkr}: Δ%", value=f"{delta:.2f}%")

                if delta > best_delta:
                    best_delta, best_tkr = delta, tkr

            except Exception as e:
                st.error(f"{tkr}: {e}")

        if best_tkr:
            st.success(f"🔮 Лучший прогноз: **{best_tkr}** (+{best_delta:.2f}%)")

    # ─── Tab 2: Индикаторы ────────────────────────────────
    with tab2:
        sel = st.selectbox("Выберите тикер", tickers)
        try:
            ohlc = load_ohlc(sel)
            # последние 3 месяца
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
            ohlc3m = ohlc.loc[ohlc.index >= cutoff].reset_index().rename(columns={"index": "Date"})

            # строим свечи
            rule = alt.Chart(ohlc3m).mark_rule(color="black").encode(
                x="Date:T",
                y="Low:Q",
                y2="High:Q"
            )
            bar = alt.Chart(ohlc3m).mark_bar().encode(
                x="Date:T",
                y="Open:Q",
                y2="Close:Q",
                color=alt.condition("datum.Close > datum.Open",
                                    alt.value("green"), alt.value("red"))
            )

            # SMA
            sma20 = ohlc3m.assign(SMA20=ohlc3m["Close"].rolling(20).mean()).dropna(subset=["SMA20"])
            sma50 = ohlc3m.assign(SMA50=ohlc3m["Close"].rolling(50).mean()).dropna(subset=["SMA50"])
            line20 = alt.Chart(sma20).mark_line(color="blue").encode(
                x="Date:T", y="SMA20:Q"
            )
            line50 = alt.Chart(sma50).mark_line(color="orange").encode(
                x="Date:T", y="SMA50:Q"
            )

            chart2 = (
                (rule + bar + line20 + line50)
                .properties(height=400)
                .interactive()
            )
            st.altair_chart(chart2, use_container_width=True)

        except Exception as e:
            st.error(f"{sel}: {e}")

    # ─── Tab 3: Портфель ─────────────────────────────────
    with tab3:
        try:
            w, er, rk, sh = suggest_portfolio(tickers)
            dfw = pd.DataFrame({"Ticker": list(w), "Weight": list(w.values())}).set_index("Ticker")
            st.table(dfw)
            st.metric("Ожидаемая доходность", f"{er:.2%}" if not np.isnan(er) else "–")
            st.metric("Риск (σ)",              f"{rk:.2%}" if not np.isnan(rk) else "–")
            st.metric("Sharpe ratio",         f"{sh:.2f}" if not np.isnan(sh) else "–")
        except Exception as e:
            st.error(f"Портфель: {e}")
