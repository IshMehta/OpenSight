import pandas as pd

data = pd.read_csv('Automated_Traffic_Volume_Counts.csv')
print('data read')
vehicle = data.groupby(['Boro', 'Yr'], as_index=False)['Vol'].sum()
print('aggregate created')
pollution = pd.read_csv("processed_pollutant_info.csv") 



pollution['Time Period'] = pollution['Time Period'].str.extract('(\d{4})')
 
pollution['Time Period'] = pd.to_numeric(pollution['Time Period'])
vehicle['Yr'] = pd.to_numeric(vehicle['Yr'])

# Merge the two DataFrames based on matching columns
merged_df = pd.merge(pollution, vehicle, left_on=['Geo Place Name', 'Time Period'], right_on=['Boro', 'Yr'], how='left')

# Display the result
columns_to_drop = ['Unnamed: 0', 'Unique ID', 'Start_Date', 'Message', 'Yr', 'Boro']
merged_df = merged_df.drop(columns=columns_to_drop)
merged_df = merged_df.rename(columns={'Vol': 'Car Count', 'Data Value': 'Emission Value'})

merged_df.to_csv('merged.csv')