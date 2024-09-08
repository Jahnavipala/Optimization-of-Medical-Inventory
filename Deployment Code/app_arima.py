import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the pre-trained ARIMA model
arima_model_path = 'arima_model.pkl'  # Ensure this path is correct
arima_model = joblib.load(arima_model_path)

# Apply the styling
st.title("Medical Forecasting with ARIMA")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the dataset
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    

    # Assuming the dataset has a column that was used to train the ARIMA model
    # e.g., 'Quantity' (modify according to your dataset)
    time_series_data = data['Quantity']  # Replace 'Quantity' with the relevant column name

    # Fit the ARIMA model on the current data
    arima_model = ARIMA(time_series_data, order=(5, 1, 0))  # Adjust (p, d, q) parameters as needed
    arima_model_fit = arima_model.fit()

    # Make predictions using the ARIMA model
    predictions = arima_model_fit.get_forecast(steps=15)
    forecasted_values = predictions.predicted_mean

    # Display the predictions
    st.subheader("Forecast for the Next 15 Weeks:")
    st.write(forecasted_values)

    # Plot the predictions
    st.subheader("ARIMA Forecast")
    plt.figure(figsize=(10, 5))
    plt.plot(forecasted_values, label='Forecast', marker='o')
    plt.title('ARIMA Forecast')
    plt.xlabel('Time')
    plt.ylabel('Quantity')
    plt.legend()
    st.pyplot(plt)
else:
    st.warning("Please upload a dataset to proceed with the forecast.")
