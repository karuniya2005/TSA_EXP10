# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 1.11.2025
### Reg No:212223240068

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------
# 1️⃣ Load the dataset
# -------------------------------------------
data = pd.read_csv('train.csv')

# Show numeric columns to pick from
print("📊 Numeric columns available:\n", data.select_dtypes(include=[np.number]).columns.tolist())

# Choose one numeric variable to simulate as a time series
target_variable = 'ram'  # You can change to 'battery_power', 'px_width', etc.

# Create a pseudo time index since this dataset has no real dates
data['Date'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
data.set_index('Date', inplace=True)

# -------------------------------------------
# 2️⃣ Plot the chosen column as a time series
# -------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(data.index, data[target_variable], color='blue')
plt.title(f'{target_variable} (Simulated Time Series)')
plt.xlabel('Date')
plt.ylabel(target_variable)
plt.grid()
plt.show()

# -------------------------------------------
# 3️⃣ Check stationarity (ADF Test)
# -------------------------------------------
def check_stationarity(series):
    result = adfuller(series.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    if result[1] < 0.05:
        print("✅ The series is stationary.\n")
    else:
        print("❌ The series is not stationary.\n")

print(f"\n📈 Checking stationarity for: {target_variable}")
check_stationarity(data[target_variable])

# -------------------------------------------
# 4️⃣ ACF & PACF Plots
# -------------------------------------------
plot_acf(data[target_variable].dropna(), lags=40)
plt.title(f'ACF Plot - {target_variable}')
plt.show()

plot_pacf(data[target_variable].dropna(), lags=40)
plt.title(f'PACF Plot - {target_variable}')
plt.show()

# -------------------------------------------
# 5️⃣ Train-Test Split (80/20)
# -------------------------------------------
train_size = int(len(data) * 0.8)
train = data[target_variable][:train_size]
test = data[target_variable][train_size:]

# -------------------------------------------
# 6️⃣ Build & Train SARIMA Model
# -------------------------------------------
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)
print(results.summary())

# -------------------------------------------
# 7️⃣ Forecasting
# -------------------------------------------
predictions = results.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# -------------------------------------------
# 8️⃣ Evaluation
# -------------------------------------------
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print(f"\n✅ Root Mean Squared Error (RMSE): {rmse:.4f}")

# -------------------------------------------
# 9️⃣ Visualization: Actual vs Predicted
# -------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, predictions, label='Predicted', color='red', linestyle='--')
plt.title(f'SARIMA Forecast for {target_variable}')
plt.xlabel('Date')
plt.ylabel(target_variable)
plt.legend()
plt.grid()
plt.show()
```


### OUTPUT:
<img width="1328" height="765" alt="image" src="https://github.com/user-attachments/assets/05b6d3b6-0648-44bf-9c7a-efd860e8c46c" />

<img width="554" height="735" alt="image" src="https://github.com/user-attachments/assets/7343534e-4930-4281-8f97-1734708e68a4" />

<img width="883" height="423" alt="image" src="https://github.com/user-attachments/assets/7ccd1482-25f4-467a-a8cc-1f5808c4d25c" />

<img width="858" height="462" alt="image" src="https://github.com/user-attachments/assets/84486d19-0f6b-485e-9f7c-b838c4b18403" />


### RESULT:
Thus the program run successfully based on the SARIMA model.
