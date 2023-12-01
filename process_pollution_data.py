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
    'Annual vehicle miles travelled (trucks)'
]

# Filter the DataFrame to include only rows with relevant indicators 

# Display the relevant data
vehicle_data_2005 = df[df['Name'].isin(relevant_indicators) & (df['Time Period'] == '2005')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})
vehicle_data_2016 = df[df['Name'].isin(relevant_indicators) & (df['Time Period'] == '2016')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})

merged_vehicle = pd.concat([vehicle_data_2005, vehicle_data_2016])


# emission_data = df[~df['Name'].isin(relevant_indicators)]



borough_rows = df[(~df['Name'].isin(relevant_indicators))] 
borough_rows = borough_rows[(df['Geo Type Name'] == 'Borough')]

borough_rows.to_csv("processed_pollutant_info.csv")





