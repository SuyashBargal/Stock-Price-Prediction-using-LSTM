import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Streamlit App Configuration
st.set_page_config(page_title='Stock Price Prediction with LSTM', layout='wide')
st.title('📈 Stock Price Prediction using LSTM Model')

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r'D:/Data Scientist BIA/Databases/all_stocks_5yr.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.dropna(inplace=True)
    return df

df = load_data()

# Sidebar for Stock Selection
stock_name = st.sidebar.selectbox('Select Stock', df['Name'].unique())

# Filter data for the selected stock
df_stock = df[df['Name'] == stock_name][['date', 'close']]
df_stock.set_index('date', inplace=True)

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
df_stock_scaled = scaler.fit_transform(df_stock)

# Create Sequences for LSTM
def create_sequences(data, time_steps=50):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(df_stock_scaled)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Model Training
if st.sidebar.button('Train Model'):
    with st.spinner('Training Model...'):
        model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
        st.success('Model Training Completed!')

# Predictions
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Evaluation Metrics
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
st.write(f'RMSE: {rmse}, MAE: {mae}')

# Plotting Results
st.subheader(f'Stock Price Prediction for {stock_name}')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_stock.index[-len(y_test):], y_test_rescaled, label='Actual Prices')
ax.plot(df_stock.index[-len(y_pred):], y_pred_rescaled, label='Predicted Prices')
ax.legend()
st.pyplot(fig)

# Show Data
if st.sidebar.checkbox('Show Raw Data', False):
    st.subheader(f'Raw Data for {stock_name}')
    st.write(df_stock.head())
