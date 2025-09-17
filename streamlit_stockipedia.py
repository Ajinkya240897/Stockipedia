
"""
Streamlit Stockipedia (single-file app)

This is the same prototype but renamed to 'stockipedia'.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import plotly.graph_objs as go

# Simple helpers (same as prototype)
def fetch_price_yfinance(ticker: str, period: str = '3y', interval: str = '1d') -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index)
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Vol_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
    df = df.dropna()
    return df

def make_features_targets(df: pd.DataFrame, lookback: int = 20, horizon: int = 5):
    df = df.copy()
    feature_cols = ['Close', 'Volume', 'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal',
                    'Momentum_10', 'RSI_14', 'ATR_14', 'Vol_20']
    X_list = []
    y_list = []
    idx = []
    for i in range(lookback, len(df) - horizon + 1):
        window = df.iloc[i - lookback:i]
        feats = []
        for col in feature_cols:
            arr = window[col].values
            feats.extend([arr[-1], arr.mean(), arr.std(), arr.min(), arr.max()])
        feats.append((df['Close'].iloc[i - 1] - df['Close'].iloc[i - 2]) / df['Close'].iloc[i - 2])
        X_list.append(feats)
        y_val = df['Close'].iloc[i + horizon - 1]
        y_list.append(y_val)
        idx.append(df.index[i + horizon - 1])
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, idx

def train_ensemble(X, y):
    gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3)
    rfr = RandomForestRegressor(n_estimators=200, max_depth=8)
    gbr.fit(X, y)
    rfr.fit(X, y)
    return {'gbr': gbr, 'rfr': rfr}

def predict_ensemble(models, X):
    p1 = models['gbr'].predict(X)
    p2 = models['rfr'].predict(X)
    preds = (p1 + p2) / 2.0
    stds = np.std(np.vstack([p1, p2]), axis=0)
    conf = 1 / (1 + (stds / (np.abs(preds) + 1e-9)))
    return preds, conf

def compute_fundamentals_score(fmp_data: dict) -> float:
    return np.nan  # placeholder for simplicity

def build_recommendation(current_price, predicted_price, conf, fundamentals_score, holding_period_days):
    implied_return = (predicted_price - current_price) / current_price * 100.0
    conf_pct = float(np.clip(conf * 100.0, 0, 100))
    if implied_return > 5 and conf_pct > 60 and fundamentals_score and fundamentals_score > 50:
        decision = 'BUY'
    elif implied_return > 0 and conf_pct > 50:
        decision = 'HOLD'
    else:
        decision = 'SELL'
    return {'decision': decision, 'implied_return_pct': round(implied_return,2), 'confidence_pct': round(conf_pct,1)}

# Streamlit UI
st.set_page_config(page_title='Stockipedia', layout='wide', initial_sidebar_state='expanded')
st.title('📈 Stockipedia')

with st.sidebar:
    st.header('Inputs')
    ticker = st.text_input('Stock symbol (e.g. AAPL, TCS). No exchange suffix', value='AAPL')
    fmp_api_key = st.text_input('FMP API Key (optional)', value='')
    cols_period = st.selectbox('Historical data period', options=['6mo', '1y', '2y', '3y', '5y'], index=3)
    interval = st.selectbox('Interval', options=['1d', '1wk'], index=0)
    horizon_choice = st.selectbox('Horizon', options=['3-15 days', '1-3 months', '3-6 months', '1-3 years'], index=1)
    use_backtest = st.checkbox('Run backtest', value=True)
    submit = st.button('Run Prediction')

horizon_map = {'3-15 days':7, '1-3 months':45, '3-6 months':120, '1-3 years':365}
horizon_days = horizon_map[horizon_choice]

if submit:
    st.write('Fetching data...')
    df = fetch_price_yfinance(ticker, period=cols_period, interval=interval)
    if df.empty:
        st.error('No data found. Try adding exchange suffix (e.g. .NS)')
    else:
        df_ind = add_technical_indicators(df)
        last_close = df_ind['Close'].iloc[-1]
        st.metric('Current Price', f"{last_close:.2f}")
        X, y, idx = make_features_targets(df_ind, lookback=30, horizon=horizon_days)
        if len(y) < 10:
            st.warning('Not enough data to train model')
        else:
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            models = train_ensemble(X_s, y)
            X_last = X_s[-1].reshape(1,-1)
            preds, conf = predict_ensemble(models, X_last)
            predicted_price = float(preds[0])
            rec = build_recommendation(last_close, predicted_price, conf[0], 50, horizon_days)
            st.metric('Predicted Price', f"{predicted_price:.2f}", delta=f"{predicted_price-last_close:+.2f}")
            st.write('Simple recommendation: ', rec['decision'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['Close'], name='Close'))
            st.plotly_chart(fig, use_container_width=True)

else:
    st.write('Enter a ticker and press Run Prediction')
