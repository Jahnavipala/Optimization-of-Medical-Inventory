import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load the pre-trained Moving Average model
ma_model_path = 'ma_model.pkl'
ma_model = joblib.load(ma_model_path)

# Apply the styling
st.title("Medical Forecasting with Moving Average")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the dataset
    data = pd.read_csv(uploaded_file)

    # Make predictions using the Moving Average model
    predictions = ma_model.predict(start=len(data), end=len(data) + 15 - 1)

    # Convert predictions to a DataFrame with a heading
    forecast_df = pd.DataFrame({
        'Week': np.arange(len(data) + 1, len(data) + 16),
        'Forecasted Quantity': predictions
    })

    # Display the predictions
    st.subheader("Forecast for the Next 15 Weeks:")
    st.write(forecast_df)

    # Plot the predictions
    st.subheader("Moving Average Forecast")
    plt.figure(figsize=(10, 5))
    plt.plot(forecast_df['Week'], forecast_df['Forecasted Quantity'], label='Forecast')
    plt.title('Moving Average Forecast')
    plt.xlabel('Week')
    plt.ylabel('Forecasted Quantity')
    plt.legend()
    st.pyplot(plt)
