import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the pre-trained SARIMAX model
with open('sarimax_model.pkl', 'rb') as file:
    model = joblib.load(file)

# Apply the styling
st.title("Medical Inventory Forecasting with SARIMAX")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the dataset
    data = pd.read_csv(uploaded_file)
    
    # Preprocess the data
    data['Dateofbill'] = pd.to_datetime(data['Dateofbill'])
    data.set_index('Dateofbill', inplace=True)
    
    # Prepare exogenous variables
    exog = data[['Final_Cost']]  # Adjust to match the number of exogenous variables used in training

    # Define forecast steps
    forecast_steps = 15
    future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(days=1), periods=forecast_steps, freq='D')
    
    # Generate future exogenous variables
    future_exog = np.zeros((forecast_steps, 1))  # Placeholder, replace with actual future values
    
    # Make predictions using the SARIMAX model
    try:
        predictions = model.get_forecast(steps=forecast_steps, exog=future_exog)
        forecasted_values = predictions.predicted_mean
        
        # Convert forecasted values to a DataFrame with headings
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted Quantity': forecasted_values
        })
        
        # Display the predictions with headings
        st.subheader("Forecast for the Next 15 Days:")
        st.write(forecast_df)
        
        # Bar Chart
        st.subheader("SARIMAX Forecast (Bar Chart)")
        plt.figure(figsize=(10, 5))
        plt.bar(forecast_df['Date'], forecast_df['Forecasted Quantity'], color='skyblue')
        plt.title('SARIMAX Forecast (Bar Chart)')
        plt.xlabel('Date')
        plt.ylabel('Forecasted Quantity')
        plt.xticks(rotation=45)
        st.pyplot(plt)
        
    except ValueError as e:
        st.error(f"Error occurred: {e}")
