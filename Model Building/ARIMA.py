import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Load the dataset
df = pd.read_csv(r"C:\Users\jahna\preprocessed_data.csv")
df['Dateofbill'] = pd.to_datetime(df['Dateofbill'])
df.set_index('Dateofbill', inplace=True)

# Split data
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Train the ARIMA model
model_arima = ARIMA(train['Quantity'], order=(1, 1, 1))
model_fit_arima = model_arima.fit()

# Make predictions
forecast_steps = len(test)
forecast_arima = model_fit_arima.get_forecast(steps=forecast_steps)
forecast_values_arima = forecast_arima.predicted_mean

# Evaluate the model
mae_arima = mean_absolute_error(test['Quantity'], forecast_values_arima)
print(f'Mean Absolute Error (MAE) for ARIMA model: {mae_arima:.2f}')

# Save the model 
joblib.dump(model_fit_arima, 'arima_model.pkl')
