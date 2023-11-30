import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# Load the CSV file into a DataFrame
file_path = 'Air_Quality.csv'
df = pd.read_csv(file_path)    


# Find the number of unique values in the specified column
#unique_names = df[target_column].unique()

# Print the result
#print(f"Unique names in '{target_column}': {unique_names}")

relevant_indicators = [
    'Annual vehicle miles traveled',
    'Annual vehicle miles travelled (cars)',
    'Annual vehicle miles travelled (trucks)',
    'Boiler Emissions- Total SO2 Emissions',
    'Boiler Emissions- Total PM2.5 Emissions',
    'Boiler Emissions- Total NOx Emissions'
]

# Filter the DataFrame to include only rows with relevant indicators 

# Display the relevant data
vehicle_data = df[df['Name'].isin(relevant_indicators) & (df['Time Period'] == '2005')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})
#print(vehicle_data)
#emission_data = df[~df['Name'].isin(relevant_indicators)]

pollution_data_2005 = df[~(df['Name'].isin(relevant_indicators)) & (df['Time Period'] == '2005')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})
pollution_data_2013 = df[~(df['Name'].isin(relevant_indicators)) & (df['Time Period'] == 'Annual Average 2013')]
pollution_data_2015 = df[~(df['Name'].isin(relevant_indicators)) & (df['Time Period'] == 'Annual Average 2015')]
pollution_data_2016 = df[~(df['Name'].isin(relevant_indicators)) & (df['Time Period'] == 'Annual Average 2016')]

# Display the filtered data
#print(pollution_data_2005)


# Merge the two DataFrames based on 'Geo Place Name' and 'Time Period'
merged_data = pd.merge(pollution_data_2005, vehicle_data, on=['Geo Place Name', 'Time Period'], how='left', suffixes=('_pollution', '_vehicle'))
#print(merged_data.columns)
test_pivot = pd.pivot_table(merged_data, values="Data Value_vehicle", index = ["Name_pollution","Geo Place Name", "Time Period" ,  "Data Value_pollution"], columns="Name_vehicle").fillna(0)
# test_pivot = pd.pivot_table(merged_data, values="Data Value_pollution", index = ["Name_pollution","Geo Place Name", "Time Period" , "Data Value_vehicle"  ], columns="Name_vehicle").fillna(0)
# test_pivot = pd.pivot(merged_data, columns=["Name_vehicle"], values="Data Value_vehicle")
# test_pivot.to_csv("pivot.csv")
#print(test_pivot)
# print(test_pivot["Data Value_pollution"])
#print(test_pivot.columns)
#merged_data.to_csv('Test.csv')
new_df = pd.read_csv("pivot.csv")
print(new_df.columns)

#preprocessing almost done. Once we have it in the correct format, we will use this boiler plate code to make our predictions:

# Specify the features and target column
feature_columns = ['Annual vehicle miles traveled', 'Annual vehicle miles travelled (cars)', 'Annual vehicle miles travelled (trucks)']
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
model.add(Dense(units=input_dim, input_dim=input_dim, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1))  # No activation function for regression

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

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