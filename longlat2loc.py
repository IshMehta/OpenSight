import pandas as pd
from geopy.geocoders import Nominatim
from shapely import wkt

# data_full = pd.read_csv("Automated_Traffic_Volume_Counts.csv")
# print("Finished Reading")
# data = data_full.head(10).copy()
# data.to_csv("subset.csv")

# print("Subset created")

data = pd.read_csv("subset.csv")
print("Finished Reading")
# Assuming df is your DataFrame with a WKT column named 'WKT_Point'
# Create a new column "Location Name" based on the WKT points

geolocator = Nominatim(user_agent="my_geocoder")

# Define a function to extract coordinates from WKT points
def extract_coordinates(row):
    point = wkt.loads(row['WktGeom'])
    return point.y, point.x  # Extracts latitude and longitude

# Apply the extract_coordinates function to create new 'Latitude' and 'Longitude' columns
data[['Latitude', 'Longitude']] = pd.DataFrame(data.apply(extract_coordinates, axis=1).tolist(), columns=['Latitude', 'Longitude'])

# Define a function for reverse geocoding
def reverse_geocode(row):
    location = geolocator.reverse((row['Latitude'], row['Longitude']), language='en')
    return location.address if location else None





# Apply the reverse_geocode function to create a new "Location Name" column
data['Location Name'] = data.apply(reverse_geocode, axis=1)

# Print the DataFrame with the new columns
print(data[['WktGeom', 'Latitude', 'Longitude', 'Location Name']])