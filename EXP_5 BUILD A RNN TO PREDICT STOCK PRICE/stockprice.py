import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

data = pd.read_csv(r"C:\Users\HP\Downloads\data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

sequence_length = 10 
x, y = [], []
for i in range(len(scaled_data) - sequence_length):
 x.append(scaled_data[i:i + sequence_length, 0])
 y.append(scaled_data[i + sequence_length, 0])
x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

model = Sequential([
 SimpleRNN(units=50, activation='relu', input_shape=(x.shape[1], 1)),
 Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, epochs=50, batch_size=32)
predictions = scaler.inverse_transform(model.predict(x))

plt.figure(figsize=(14, 7))
plt.plot(data.index[sequence_length:], data['Close'][sequence_length:], label='Actual Prices')
plt.plot(data.index[sequence_length:], predictions, label='Predicted Prices')
plt.title('Stock Price Prediction Using RNN')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
