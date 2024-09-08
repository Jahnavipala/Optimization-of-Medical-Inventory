import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained LSTM model
with open('lstm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Apply the styling
st.title("Medical Inventory Forecasting with LSTM")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the dataset
    data = pd.read_csv(uploaded_file)
    
    # Preprocess the data
    data['Dateofbill'] = pd.to_datetime(data['Dateofbill'])
    data.set_index('Dateofbill', inplace=True)
    data = data[['Quantity']].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Prepare data for prediction
    time_step = 10
    
    def create_dataset(data, time_step=1):
        X = []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
        return np.array(X)
    
    forecast_input = create_dataset(scaled_data, time_step)
    forecast_input = forecast_input.reshape((forecast_input.shape[0], forecast_input.shape[1], 1))
    
    # Make predictions
    y_pred = model.predict(forecast_input[-1].reshape(1, time_step, 1))
    
    # Inverse transform the predictions
    y_pred = scaler.inverse_transform(y_pred)
    
    # Display the predictions
    st.subheader("Forecast for the Next Time Step:")
    st.write(y_pred[0, 0])
    
    # Plot the predictions
    st.subheader("LSTM Forecast")
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(scaled_data)), scaler.inverse_transform(scaled_data), label='Actual Data')
    plt.axhline(y=y_pred[0, 0], color='r', linestyle='--', label='Forecast')
    plt.title('LSTM Forecast vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Quantity')
    plt.legend()
    st.pyplot(plt)
