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

# Define and train the SARIMA model
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
model_sarima = SARIMAX(train['Quantity'], order=order, seasonal_order=seasonal_order)
model_fit_sarima = model_sarima.fit()

# Make predictions
forecast_steps = len(test)
forecast_sarima = model_fit_sarima.get_forecast(steps=forecast_steps)
forecast_values_sarima = forecast_sarima.predicted_mean

# Evaluate the model
mae_sarima = mean_absolute_error(test['Quantity'], forecast_values_sarima)
print(f'Mean Absolute Error (MAE) for SARIMA model: {mae_sarima:.2f}')

# Save the model
joblib.dump(model_fit_sarima, 'sarima_model.pkl')
