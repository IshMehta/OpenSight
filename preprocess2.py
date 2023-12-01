import pandas as pd

data = pd.read_csv("Air_Quality.csv")

borough_mapping = {
    'Bronx': ['Fordham - Bronx Pk', 'High Bridge - Morrisania', 'Northeast Bronx', 'Hunts Point - Mott Haven', 'Morrisania and Crotona (CD3)', 'Bronx', 'Hunts Point and Longwood (CD2)', 'South Bronx', 'Throgs Neck and Co-op City (CD10)', 'Williamsbridge and Baychester (CD12)', 'Jamaica and Hollis (CD12)', 'Crotona -Tremont', 'Highbridge and Concourse (CD4)', 'Fordham and University Heights (CD5)', 'Belmont and East Tremont (CD6)', 'Parkchester and Soundview (CD9)', 'Morris Park and Bronxdale (CD11)', 'Brownsville (CD16)'],
    'Brooklyn': ['Bedford Stuyvesant - Crown Heights', 'Bensonhurst - Bay Ridge', 'Greenpoint', 'Coney Island - Sheepshead Bay', 'Williamsburg - Bushwick', 'Central Harlem - Morningside Heights', 'Downtown - Heights - Slope', 'Kingsbridge - Riverdale', 'Greenwich Village - SoHo', 'Bedford Stuyvesant (CD3)', 'Bushwick (CD4)', 'Ridgewood and Maspeth (CD5)', 'Sunset Park (CD7)', 'Morningside Heights and Hamilton Heights (CD9)', 'South Crown Heights and Lefferts Gardens (CD9)', 'Union Square-Lower Manhattan', 'Fort Greene and Brooklyn Heights (CD2)', 'Park Slope and Carroll Gardens (CD6)', 'Crown Heights and Prospect Heights (CD8)', 'Flatbush and Midwood (CD14)'],
    'Manhattan': ['Chelsea-Village', 'Financial District (CD1)', 'Greenwich Village and Soho (CD2)', 'Woodside and Sunnyside (CD2)', 'Lower East Side and Chinatown (CD3)', 'Upper East Side (CD8)', 'Central Harlem (CD10)', 'Washington Heights and Inwood (CD12)', 'Lower Manhattan', 'Midtown (CD5)', 'Chelsea - Clinton', 'Union Square - Lower East Side', 'Union Square-Lower Manhattan', 'New York City'],
    'Queens': ['Borough Park', 'Rockaways', 'Greenpoint and Williamsburg (CD1)', 'Queens Village (CD13)', 'Queens', 'Long Island City - Astoria', 'Flushing - Clearview', 'Southeast Queens', 'Upper East Side-Gramercy', 'Bayside Little Neck-Fresh Meadows', 'Rockaway and Broad Channel (CD14)', 'Jamaica', 'Sunset Park', 'Bayside and Little Neck (CD11)', 'South Beach and Willowbrook (CD2)', 'Staten Island', 'South Beach - Tottenville', 'Flushing and Whitestone (CD7)', 'Kew Gardens and Woodhaven (CD9)', 'Bayside and Little Neck (CD11)', 'Queens Village (CD13)', 'New York City'],
    'Staten Island': ['Northern SI', 'Port Richmond', 'St. George and Stapleton (CD1)', 'Tottenville and Great Kills (CD3)', 'Willowbrook'],
}

# Reverse mapping for easy lookup
reverse_borough_mapping = {location: borough for borough, locations in borough_mapping.items() for location in locations}

# Apply the mapping to create a new "Borough" column
data['Borough'] = data['Geo Place Name'].map(reverse_borough_mapping)

data.to_csv("test2.csv")
