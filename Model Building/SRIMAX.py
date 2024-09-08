import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# Load the dataset
df = pd.read_csv(r"C:\Users\jahna\preprocessed_data.csv")
df['Dateofbill'] = pd.to_datetime(df['Dateofbill'])
df.set_index('Dateofbill', inplace=True)

# Split data
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Define and train the SARIMAX model
exog = df[['Final_Cost']]  # Example of exogenous variables
model_sarimax = SARIMAX(train['Quantity'], exog=exog[:train_size], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit_sarimax = model_sarimax.fit()

# Make predictions
exog_test = df[['Final_Cost']][train_size:]
forecast_steps = len(test)
forecast_sarimax = model_fit_sarimax.get_forecast(steps=forecast_steps, exog=exog_test)
forecast_values_sarimax = forecast_sarimax.predicted_mean

# Evaluate the model
mae_sarimax = mean_absolute_error(test['Quantity'], forecast_values_sarimax)
print(f'Mean Absolute Error (MAE) for SARIMAX model: {mae_sarimax:.2f}')

# Save the model
joblib.dump(model_fit_sarimax, 'sarimax_model.pkl')
