import pandas as pd

data = pd.read_csv("pivot.csv")
print(data["Geo Place Name"].unique())

data = data.replace("East Harlem (CD11)", "East Harlem")
data = data.replace("East New York and Starrett City (CD5)", "East New York")
data = data.replace("Elmhurst and Corona (CD4)", "Elmhurst and Corona")

print(data["Geo Place Name"].unique())

