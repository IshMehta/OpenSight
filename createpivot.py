import pandas as pd 

# Load the CSV file into a DataFrame
file_path = 'Air_Quality.csv'
df = pd.read_csv(file_path)    

relevant_indicators = [
    'Annual vehicle miles traveled',
    'Annual vehicle miles travelled (cars)',
    'Annual vehicle miles travelled (trucks)',
    'Boiler Emissions- Total SO2 Emissions',
    'Boiler Emissions- Total PM2.5 Emissions',
    'Boiler Emissions- Total NOx Emissions'
]

# Filter the DataFrame to include only rows with relevant indicators 

#create vehicle data data frame
vehicle_data_2005 = df[df['Name'].isin(relevant_indicators) & (df['Time Period'] == '2005')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})
vehicle_data_2016 = df[df['Name'].isin(relevant_indicators) & (df['Time Period'] == '2016')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})
merged_vehicle = pd.concat([vehicle_data_2005, vehicle_data_2016]) 

#create pollution data dataframe
pollution_data_2005 = df[~(df['Name'].isin(relevant_indicators)) & (df['Time Period'] == '2005')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})
pollution_data_2016 = df[~(df['Name'].isin(relevant_indicators)) & (df['Time Period'] == 'Annual Average 2016')].groupby(['Name', 'Geo Place Name', 'Time Period'], as_index=False).agg({'Data Value': 'mean'})
pollution_data_2016 = pollution_data_2016.replace("Annual Average 2016", "2016")


merged_pollution = pd.concat([pollution_data_2005, pollution_data_2016]) 


# Merge the two DataFrames based on 'Geo Place Name' and 'Time Period'
merged_data = pd.merge(merged_pollution, merged_vehicle, on=['Geo Place Name', 'Time Period'], how='left', suffixes=('_pollution', '_vehicle'))
test_pivot = pd.pivot_table(merged_data, values="Data Value_vehicle", index = ["Name_pollution","Geo Place Name", "Time Period" ,  "Data Value_pollution"], columns="Name_vehicle").fillna(0)
test_pivot.to_csv("pivot.csv") 


