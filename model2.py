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
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import add
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tensorflow import keras

import shap
# print the JS visualization code to the notebook
shap.initjs()


new_df = pd.read_csv("pivot.csv")
# print(new_df.columns)

# print(new_df["Geo Place Name"].unique())

new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Bay Ridge and Dyker Heights (CD10)", "Bay Ridge and Dyker Heights")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Bayside - Little Neck", "Bayside and Little Neck")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Bayside and Little Neck (CD11)", "Bayside and Little Neck")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Bedford Stuyvesant (CD3)", "Bedford Stuyvesant")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Bedford Stuyvesant - Crown Heights", "Bedford Stuyvesant")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Belmont and East Tremont (CD6)", "Belmont and East Tremont")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Bensonhurst (CD11)", "Bensonhurst")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Bensonhurst - Bay Ridge", "Bensonhurst")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Borough Park (CD12)", "Borough Park")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Canarsie - Flatlands", "Flatlands")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Central Harlem (CD10)", "Central Harlem")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Central Harlem - Morningside Heights", "Central Harlem")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Chelsea - Clinton", "Chelsea and Clinton")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Clinton and Chelsea (CD4)", "Chelsea and Clinton")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Coney Island (CD13)", "Coney Island")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Coney Island - Sheepshead Bay", "Coney Island")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Crotona -Tremont", "Crotona")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("East Flatbush (CD17)", "East Flatbush")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("East Flatbush - Flatbush", "East Flatbush")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("East Harlem (CD11)", "East Harlem")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("East New York and Starrett City (CD5)", "East New York")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Flatlands and Canarsie (CD18)", "Flatlands")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Flushing - Clearview", "Flushing")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Flushing and Whitestone (CD7)", "Flushing")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Fordham - Bronx Pk", "Fordham")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Fordham and University Heights (CD5)", "Fordham")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Greenpoint and Williamsburg (CD1)", "Greenpoint")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Greenwich Village - SoHo", "Greenwich Village and Soho")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Greenwich Village and Soho (CD2)", "Greenwich Village and Soho")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("High Bridge - Morrisania", "High Bridge")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Highbridge and Concourse (CD4)", "High Bridge")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Hillcrest and Fresh Meadows (CD8)", "Fresh Meadows")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Hunts Point - Mott Haven", "Hunts Point")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Hunts Point and Longwood (CD2)", "Hunts Point")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Jamaica and Hollis (CD12)", "Jamaica")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Kingsbridge - Riverdale", "Kingsbridge")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Kingsbridge Heights and Bedford (CD7)", "Kingsbridge")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Long Island City - Astoria", "Long Island City")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Long Island City and Astoria (CD1)", "Long Island City")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Morrisania and Crotona (CD3)", "Crotona")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Northeast Bronx", "Bronx")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Queens Village (CD13)", "Queens")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Rego Park and Forest Hills (CD6)", "Forest Hills")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Ridgewood - Forest Hills", "Forest Hills")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Rockaway and Broad Channel (CD14)", "Rockaways")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("South Beach - Tottenville", "South Beach")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("South Beach and Willowbrook (CD2)", "South Beach")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Southeast Queens", "Queens")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Southwest Queens", "Queens")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("St. George and Stapleton (CD1)", "St. George")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Stapleton - St. George", "St. George")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Sunset Park (CD7)", "Sunset Park")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Upper East Side (CD8)", "Upper East Side")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Upper West Side (CD7)", "Upper West Side")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Washington Heights and Inwood (CD12)", "Washington Heights")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Williamsbridge and Baychester (CD12)", "Williamsbridge")
new_df["Geo Place Name"] = new_df["Geo Place Name"].replace("Williamsburg - Bushwick", "Williamsbridge")

# print(new_df["Geo Place Name"].unique())









# new_df = pd.read_csv("merged.csv")
# print(new_df.columns)

# Create Location mapping
loc_map = {}
loc_names = new_df["Geo Place Name"].unique()
for i in range(len(loc_names)):
    loc_map[loc_names[i]] = i + 1

# # Create Name Mapping
# name_map = {}
# names = new_df["Name"].unique()
# for i in range(len(names)):
#     name_map[names[i]] = i + 1

new_df["Geo Place Name"] = new_df["Geo Place Name"].map(loc_map)
# new_df["Name"] = new_df["Name"].map(name_map)


# fill in nans
# new_df = new_df.fillna(0)
# new_df = new_df.dropna()

# new_df.to_csv("final_preprocess.csv")


#################################################################################
#
# Neural Network
#
#################################################################################
#preprocessing almost done. Once we have it in the correct format, we will use this boiler plate code to make our predictions:

# Specify the features and target column
# for pivot.csv
feature_columns = ['Annual vehicle miles traveled', 'Annual vehicle miles travelled (cars)', 'Annual vehicle miles travelled (trucks)', 'Time Period', 'Geo Place Name']
target_column = 'Data Value_pollution'

# # feature_columns = ['Name', 'Geo Place Name', 'Time Period', "Car Count"]
# # target_column = 'Emission Value'


# Extract features and target
X = new_df[feature_columns].astype(float)
y = new_df[target_column].astype(float)

# print(X.isna().any().any())
# print(y.isna().any().any())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # Define a basic residual block for fully connected layers
# def residual_block(x, units):
#     # Shortcut connection
#     shortcut = x
    
#     # First fully connected layer
#     x = Dense(units)(x)
#     x = BatchNormalization()(x)
#     x = Activation('leaky_relu')(x)
    
#     # Second fully connected layer
#     x = Dense(units)(x)
#     x = BatchNormalization()(x)
    
#     # Add the shortcut to the output
#     x = add([x, shortcut])
#     x = Activation('leaky_relu')(x)
    
#     return x


# Specify the output layer for regression
model = Sequential()
input_dim = X_train_scaled.shape[1]
model.add(Dense(units=input_dim, input_dim=input_dim, activation='leaky_relu'))

# Add residual block
# model.add(residual_block(model.layers[-1].output, input_dim))

# Additional fully connected layers with residual blocks
# model.add(residual_block(model.layers[-1].output, 10))
# model.add(residual_block(model.layers[-1].output, 10))
model.add(BatchNormalization(synchronized=True))
model.add(Dense(units=5000, activation='leaky_relu'))
model.add(BatchNormalization(synchronized=True))
model.add(Dense(units=2000, activation='leaky_relu'))
model.add(BatchNormalization(synchronized=True))
model.add(Dense(units=1000, activation='leaky_relu'))
model.add(BatchNormalization(synchronized=True))
model.add(Dense(units=500, activation='leaky_relu'))
model.add(BatchNormalization(synchronized=True))
model.add(Dense(units=50, activation='leaky_relu'))
model.add(BatchNormalization(synchronized=True))


# model.add(Dense(units=200, activation='leaky_relu'))
# # model.add(BatchNormalization(synchronized=True))
# model.add(Dense(units=100, activation='leaky_relu'))

# model.add(BatchNormalization(synchronized=True))
model.add(Dense(units=1))  # No activation function for regression






# Compile and train the model
# opt = keras.optimizers.Adam(learning_rate = 0.0005)
opt = keras.optimizers.SGD(learning_rate = 0.00005)
model.compile(optimizer=opt, loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=200, batch_size=8, validation_split=0.2, verbose=2)

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

#################################################################################
#
# SHAP
#
#################################################################################

