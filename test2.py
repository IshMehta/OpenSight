import pandas as pd

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
print(vehicle_data)
#emission_data = df[~df['Name'].isin(relevant_indicators)]

pollution_data_2005 = df[~(df['Name'].isin(relevant_indicators)) & (df['Time Period'] == '2005')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})
pollution_data_2013 = df[~(df['Name'].isin(relevant_indicators)) & (df['Time Period'] == 'Annual Average 2013')]
pollution_data_2015 = df[~(df['Name'].isin(relevant_indicators)) & (df['Time Period'] == 'Annual Average 2015')]
pollution_data_2016 = df[~(df['Name'].isin(relevant_indicators)) & (df['Time Period'] == 'Annual Average 2016')]

# Display the filtered data
#print(pollution_data_2005)


# Merge the two DataFrames based on 'Geo Place Name' and 'Time Period'
merged_data = pd.merge(pollution_data_2005, vehicle_data, on=['Geo Place Name', 'Time Period'], how='left', suffixes=('_pollution', '_vehicle'))

# Update the 'Data Value' column in 'pollution_data_2005' with values from 'vehicle_data'
pollution_data_2005['Data Value'] = merged_data['Data Value_vehicle'].fillna(merged_data['Data Value_pollution'])

# Drop the unnecessary columns from the merged data 

# Display the updated 'pollution_data_2005'
print(merged_data)

# condensed_data = merged_data.pivot_table(index=['Geo Place Name', 'Time Period'], columns='Name_vehicle', values=['Data Value_pollution', 'Data Value_vehicle'], aggfunc='first')

# # Reset the index to flatten the multi-level columns
# condensed_data.reset_index(inplace=True)

# # Display the condensed data
# condensed_data.to_csv("test_3.csv")

# print(sorted(emission_data['Time Period'].unique())) 


# emission_data = emission_data.groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})
# vehicle_data = vehicle_data.groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})

# vehicle_data.to_csv('vehicle_data.csv')
# emission_data.to_csv('emission_data.csv')


