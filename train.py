import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import json

data = pd.read_csv("car_price_prediction.csv")
# Drop unnecessary columns
data.drop(["ID"], axis=1)
# Drop duplicate rows
data.drop_duplicates(inplace=True)

num_features = ["Levy", "Prod. year", "Mileage", "Cylinders", "Airbags"]
nom_features = ["Model", "Manufacturer", "Category", "Fuel type", "Gear box type", "Drive wheels", "Doors", "Color"]
ord_features = ["Wheel", "Leather interior"]
# Data preprocessing
data["Levy"] = data["Levy"].replace('-', '0').astype(int)
data["Engine volume"] = data["Engine volume"].apply(lambda x: float(x.split(" ")[0]))
data["Mileage"] = data["Mileage"].apply(lambda x: float(x.split(" ")[0]))

value_counts = data["Model"].value_counts()
values_to_map = value_counts[value_counts <= 10].index

data['Model'] = data['Model'].apply(lambda x: "Other" if x in values_to_map else x)

# Remove outliers
def iqr_remove_outliers(data, col):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

for i in num_features:
    data = iqr_remove_outliers(data, i)
data = iqr_remove_outliers(data, "Price")

# Save unique values
unique_values_dict = {}
for i in nom_features + ord_features:
    unique_values_dict[i] = data[i].unique().tolist()
with open("unique_value_columns.json", "w") as f:
    f.write(json.dumps(unique_values_dict))
    
# Split data into features (x) and target (y)
target = "Price"
x = data.drop(target, axis=1)
y = data[target]
# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# Preprocessing and modeling pipeline
preprocessor = ColumnTransformer(transformers=[
    ("numeric_features", StandardScaler(), num_features),
    ("ordinal_features", OrdinalEncoder(categories=[x_train[i].unique() for i in ord_features]), ord_features),
    ("nominal_features", OneHotEncoder(sparse=False, handle_unknown='ignore'), nom_features),
])
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42, n_estimators=30)),
])

# Fit the model and make predictions
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Evaluate the model's performance
print(f"r2_score: {r2_score(y_test, y_pred)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
# Visualization: Residual plot
plt.figure(figsize=(8, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, color='blue', alpha=0.6)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
# plt.show()
plt.savefig("residualplot.jpg")
# Save model
with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)









