import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title('Nifty 50 Stock Price Prediction')

# User input
ticker = st.text_input('Enter Stock Ticker', 'RELIANCE.NS')
start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2023-01-01'))

# Fetch data
data = yf.download(ticker, start=start_date, end=end_date)
if data.empty:
    st.error("No data available for the selected ticker and date range.")
else:
    st.write(data)

    # Prepare the data
    data['Date'] = pd.to_datetime(data.index)
    data['Date'] = data['Date'].map(pd.Timestamp.toordinal)
    X = data[['Date']]
    y = data['Close']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future prices
    if st.button('Predict'):
        future_dates = pd.date_range(end_date, periods=30).map(pd.Timestamp.toordinal).reshape(-1, 1)
        future_predictions = model.predict(future_dates)
        st.write(f'Predicted Prices for the next 30 days: {future_predictions}')
