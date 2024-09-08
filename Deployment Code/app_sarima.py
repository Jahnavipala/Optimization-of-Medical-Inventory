import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the pre-trained SARIMA model
sarima_model_path = 'sarima_model.pkl'
sarima_model = joblib.load(sarima_model_path)

# Apply the styling
st.title("Medical Forecasting with SARIMA")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the dataset
    data = pd.read_csv(uploaded_file)

    # Make predictions using the SARIMA model
    predictions = sarima_model.get_forecast(steps=15)
    forecasted_values = predictions.predicted_mean

    # Convert forecasted values to a DataFrame with headings
    forecast_df = pd.DataFrame({
        'Week': np.arange(1, 16),  # Weeks 1 to 15
        'Forecasted Quantity': forecasted_values
    })

    # Display the predictions with headings
    st.subheader("Forecast for the Next 15 Weeks:")
    st.write(forecast_df)

    # Plot the predictions
    st.subheader("SARIMA Forecast")
    plt.figure(figsize=(10, 5))
    plt.plot(forecast_df['Week'], forecast_df['Forecasted Quantity'], label='Forecast', marker='o')
    plt.title('SARIMA Forecast')
    plt.xlabel('Week')
    plt.ylabel('Forecasted Quantity')
    plt.legend()
    st.pyplot(plt)
else:
    st.warning("Please upload a dataset to proceed with the forecast.")

