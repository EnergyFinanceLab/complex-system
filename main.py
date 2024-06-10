import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define the input data
data = np.array([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15]])

# Reshape the data to fit LSTM input shape
data = np.reshape(data, (data.shape[0], data.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(data.shape[1], 1)))
model.add(Dense(1))
# Add the model training code
model.fit(data, data, epochs=100, batch_size=1))

# Generate predictions
predictions = model.predict(data)

# Print the predictions
print(predictions)