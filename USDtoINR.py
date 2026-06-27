import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Download USD/INR data
data = yf.download("INR=X", start="2023-01-01", progress=False)

# Use closing prices
data = data[['Close']].copy()

# Create ML features
data['Prev_Close'] = data['Close'].shift(1)
data['MA_5'] = data['Close'].rolling(5).mean()
data['MA_10'] = data['Close'].rolling(10).mean()
data['MA_20'] = data['Close'].rolling(20).mean()

# Remove missing values
data.dropna(inplace=True)

# Features and target
X = data[['Prev_Close', 'MA_5', 'MA_10', 'MA_20']]
y = data['Close']

# Train/Test split
split = int(len(data) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# Random Forest Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test predictions
predictions = model.predict(X_test)

# Accuracy metric
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", round(mae, 4))

# Predict tomorrow
latest = X.iloc[[-1]]
tomorrow = model.predict(latest)

print("\nPredicted USD/INR for Tomorrow:")
print(round(float(tomorrow[0]), 4))

# Graph
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Random Forest USD/INR Prediction")
plt.xlabel("Days")
plt.ylabel("Exchange Rate")
plt.grid(True)
plt.show()