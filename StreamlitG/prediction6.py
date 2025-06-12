# Good prediction for 12 hours / batch 32 epoch 6
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping
from keras.models import load_model
import joblib

# Constants
DATA_PATH = "train_cleaned_dataset_modified.csv"
n_past = 7 * 24  # 7 days * 24 hours
n_future = 12  # 12 hours
split_ratio = 0.8

# Load dataset
df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col=[0])

# Select columns
input_columns = ["CH4_w-out", "CH4_s-out", "CH4_n-out", "CH4_e-out", "CH4_n-in", "CH4_m-in", "CH4_m-in-up", "CH4_s-in", "CH4_w-in", "CH4_e-in", "TEMP", "Ver_w", "Hor_w", "CH4_in_mean", "CH4_out_mean","CO2_n-in","CO2_m-in","CO2_m-in-up","CO2_s-in","CO2_w-in","CO2_e-in"]
target_columns = ["CH4_in_mean"]

# Prepare data
data = df[input_columns].values
target_data = df[target_columns].values

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_data_scaled = target_scaler.fit_transform(target_data)

# Save the scalers
joblib.dump(scaler, "scaler.pkl")
joblib.dump(target_scaler, "target_scaler.pkl")

# Function to create dataset
def create_dataset(dataset, target, n_past, n_future):
    dataX, dataY = [], []
    for i in range(len(dataset) - n_past - n_future + 1):
        dataX.append(dataset[i:(i + n_past), :])
        dataY.append(target[(i + n_past):(i+ n_past + n_future), :])
    return np.array(dataX), np.array(dataY)

# Generate datasets
X, Y = create_dataset(data_scaled, target_data_scaled, n_past, n_future)

# Split data into train/test
split_point = int(len(X) * split_ratio)
trainX, testX = X[:split_point], X[split_point:]
trainY, testY = Y[:split_point], Y[split_point:]

# Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(n_past, len(input_columns))))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(50, activation='relu'))
model.add(Dense(trainY.shape[1]))
model.compile(loss='mse', optimizer='adam', metrics=['mae'], run_eagerly=True)

model.summary()

# Define early stopping callback
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit model with early stopping
history = model.fit(trainX, trainY, epochs=1 , batch_size=128, validation_data=(testX, testY), verbose=1, callbacks=[early_stopping_callback])




# Save the model
model.save("model.h5")


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Predict and inverse transform the prediction
predicted = model.predict(testX)
predicted_inversed = target_scaler.inverse_transform(predicted)

# Flatten testY to align with the predicted values
true_values_flattened = testY.reshape(-1, 1)
true_values_aligned = target_scaler.inverse_transform(true_values_flattened)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(true_values_aligned, label='True', alpha=0.7)
predicted_values = predicted_inversed.flatten()
plt.plot(predicted_values, label='Predicted', alpha=0.7)
plt.title('CH4_in_mean Predictions vs True Values')
plt.legend()
plt.show()

# Plotting the results with 4 subplots for the first 4 sequences of 12 predictions
fig, axs = plt.subplots(4, 1, figsize=(10, 20))  # 4 rows, 1 column

for i in range(4):
    # Indices for the current sequence of 12 predictions
    start_idx = i * 12
    end_idx = start_idx + 12
    
    # Select the current sequence for true and predicted values
    true_seq = true_values_aligned[start_idx:end_idx]
    pred_seq = predicted_values[start_idx:end_idx]
    
    # Plotting the current sequence
    axs[i].plot(true_seq, label='True Values', marker='o', linestyle='-', color='blue')
    axs[i].plot(pred_seq, label='Predicted Values', marker='x', linestyle='--', color='red')
    axs[i].set_title(f'Sequence {i+1}')
    axs[i].legend()

plt.tight_layout()
plt.show()


# Initialize an empty list to hold all predicted dates
predicted_dates = []
for i in range(len(testX)):
    actual_start_index = split_point + i 
    start_date = df.index[actual_start_index]
    for j in range(n_future):
        predicted_date = start_date + pd.Timedelta(hours=j)
        predicted_dates.append(predicted_date)
        
predicted_flat = predicted_inversed.flatten()
# Create a DataFrame for the predictions with dates
predicted_withDate_Flat = pd.DataFrame({
    'Date': predicted_dates,
    'Prediction': predicted_flat
})


print("Hi")