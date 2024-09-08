import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"C:\Users\jahna\preprocessed_data.csv")
df['Dateofbill'] = pd.to_datetime(df['Dateofbill'])
df.set_index('Dateofbill', inplace=True)

# Feature Engineering
df['Month'] = df.index.month
df['Day'] = df.index.day
df['Year'] = df.index.year

# Define features and target
X = df[['Month', 'Day', 'Year']]
y = df['Quantity']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define and train the Random Forest model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Make predictions
y_pred = model_rf.predict(X_test)

# Evaluate the model
mae_rf = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE) of Random Forest model: {mae_rf}")

# Save the model 
joblib.dump(model_rf, 'random_forest_model.pkl')
