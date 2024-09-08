import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error

# Load the dataset
df = pd.read_csv(r"C:\Users\jahna\preprocessed_data.csv")
df['Dateofbill'] = pd.to_datetime(df['Dateofbill'])
df.set_index('Dateofbill', inplace=True)

# Split data
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Train the Moving Average model
model_ma = SimpleExpSmoothing(train['Quantity'])
model_fit_ma = model_ma.fit()

# Make predictions
forecast_steps = len(test)
forecast_ma = model_fit_ma.forecast(steps=forecast_steps)

# Evaluate the model
mae_ma = mean_absolute_error(test['Quantity'], forecast_ma)
print(f'Mean Absolute Error (MAE) for MA model: {mae_ma:.2f}')

# Save the model
joblib.dump(model_fit_ma, 'ma_model.pkl')