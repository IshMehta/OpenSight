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
    'Annual vehicle miles travelled (trucks)',
    'Boiler Emissions- Total SO2 Emissions',
    'Boiler Emissions- Total PM2.5 Emissions',
    'Boiler Emissions- Total NOx Emissions'
]

# Filter the DataFrame to include only rows with relevant indicators 

# Display the relevant data
vehicle_data_2005 = df[df['Name'].isin(relevant_indicators) & (df['Time Period'] == '2005')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})
vehicle_data_2016 = df[df['Name'].isin(relevant_indicators) & (df['Time Period'] == '2016')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})

merged_vehicle = pd.concat([vehicle_data_2005, vehicle_data_2016])


# emission_data = df[~df['Name'].isin(relevant_indicators)]

pollution_data_2005 = df[~(df['Name'].isin(relevant_indicators)) & (df['Time Period'] == '2005')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})
pollution_data_2013 = df[~(df['Name'].isin(relevant_indicators)) & (df['Time Period'] == 'Annual Average 2013')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})
pollution_data_2013 = pollution_data_2013.replace("Annual Average 2013", "2013")
pollution_data_2015 = df[~(df['Name'].isin(relevant_indicators)) & (df['Time Period'] == 'Annual Average 2015')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})
pollution_data_2015 = pollution_data_2015.replace("Annual Average 2015", "2015")
pollution_data_2016 = df[~(df['Name'].isin(relevant_indicators)) & (df['Time Period'] == 'Annual Average 2016')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})
pollution_data_2016 = pollution_data_2016.replace("Annual Average 2016", "2016")


merged_pollution = pd.concat([pollution_data_2005, pollution_data_2016])

# Display the filtered data
print(pollution_data_2005)


# Merge the two DataFrames based on 'Geo Place Name' and 'Time Period'
merged_data = pd.merge(merged_pollution, merged_vehicle, on=['Geo Place Name', 'Time Period'], how='left', suffixes=('_pollution', '_vehicle'))
#print(merged_data.columns)
test_pivot = pd.pivot_table(merged_data, values="Data Value_vehicle", index = ["Name_pollution","Geo Place Name", "Time Period" ,  "Data Value_pollution"], columns="Name_vehicle").fillna(0)
# test_pivot = pd.pivot_table(merged_data, values="Data Value_pollution", index = ["Name_pollution","Geo Place Name", "Time Period" , "Data Value_vehicle"  ], columns="Name_vehicle").fillna(0)
# test_pivot = pd.pivot(merged_data, columns=["Name_vehicle"], values="Data Value_vehicle")
test_pivot.to_csv("pivot.csv")
#print(test_pivot)
# print(test_pivot["Data Value_pollution"])
#print(test_pivot.columns)
merged_data.to_csv('Test.csv')



