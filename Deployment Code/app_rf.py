import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load the pre-trained Random Forest model
rf_model_path = 'random_forest_model.pkl'
rf_model = joblib.load(rf_model_path)

# Apply the styling
st.title("Medical Inventory Forecasting with Random Forest")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the dataset
    data = pd.read_csv(uploaded_file)

    # Feature Engineering for future dates
    future_dates = pd.date_range(start=data['Dateofbill'].max(), periods=16, freq='D')[1:]
    future_df = pd.DataFrame({'Dateofbill': future_dates})
    future_df['Month'] = future_df['Dateofbill'].dt.month
    future_df['Day'] = future_df['Dateofbill'].dt.day
    future_df['Year'] = future_df['Dateofbill'].dt.year

    # Make predictions using the Random Forest model
    future_features = future_df[['Month', 'Day', 'Year']]
    forecasted_quantities = rf_model.predict(future_features)

    # Display the predictions
    st.subheader("Forecasted Quantities for the Next 15 Days:")
    forecast_results = future_df.copy()
    forecast_results['Predicted Quantity'] = forecasted_quantities
    st.write(forecast_results)

    # Plot the predictions
    st.subheader("Random Forest Forecast")
    plt.figure(figsize=(10, 5))
    plt.plot(future_df['Dateofbill'], forecasted_quantities, label='Forecasted Quantity', marker='o')
    plt.title('Random Forest Forecast for Future Quantities')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    st.pyplot(plt)
