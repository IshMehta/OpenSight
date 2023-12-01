import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


new_df = pd.read_csv("pivot.csv")
print(new_df.columns)


#################################################################################
#
# Neural Network
#
#################################################################################
#preprocessing almost done. Once we have it in the correct format, we will use this boiler plate code to make our predictions:

# Specify the features and target column
feature_columns = ['Annual vehicle miles traveled', 'Annual vehicle miles travelled (cars)', 'Annual vehicle miles travelled (trucks)', 'Time Period']
target_column = 'Data Value_pollution'

# Extract features and target
X = new_df[feature_columns]
y = new_df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Specify the output layer for regression
model = Sequential()
input_dim = X_train_scaled.shape[1]
model.add(Dense(units=input_dim, input_dim=input_dim, activation='leaky_relu'))
model.add(Dense(units=200, activation='leaky_relu'))
# model.add(BatchNormalization(synchronized=True))
# model.add(Dense(units=500, activation='leaky_relu'))
model.add(Dense(units=200, activation='leaky_relu'))
# model.add(BatchNormalization(synchronized=True))
model.add(Dense(units=100, activation='leaky_relu'))

# model.add(BatchNormalization(synchronized=True))
model.add(Dense(units=1))  # No activation function for regression

# Compile and train the model
model.compile(optimizer='SGD', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=200, batch_size=64, validation_split=0.2, verbose=2)

# Make predictions
predictions = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

r_squared = r2_score(y_test, predictions)
print(f'R-squared: {r_squared}')

plt.plot(y_test.values, label='Actual Values', linestyle='-', marker='o')
plt.plot(predictions, label='Predicted Values', linestyle='-', marker='o')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()