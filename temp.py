import tensorflow as tf
from tensorflow.keras import layers, models
#from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print(data.shape[1])
print(target.shape)
print(data)
# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print(data)

run = 0
if run == 1:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.2, random_state=42)

    # Define the model architecture
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print('Test MAE:', test_mae)




