import json
import pickle

import numpy as np
# import numpy as np
import pandas as pd

with open('unique_value_columns.json', 'r') as f:
    unique_values_dict = json.load(f)

with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

features = ['Levy', 'Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior',
            'Fuel type', 'Engine volume', 'Mileage', 'Cylinders', 'Gear box type', 'Drive wheels',
            'Doors', 'Wheel', 'Color', 'Airbags']
num_features = ["Levy", "Prod. year", "Mileage", 'Engine volume', "Cylinders", "Airbags"]

test_data = []
for i in features:
    if i in num_features:
        x = int(input(f"{i}: "))
    else:
        max_len = max(len(value) for value in unique_values_dict[i])
        for j, value in enumerate(unique_values_dict[i], start=1):
            if j % 5 != 0:
                print(f"{j}.{value}".ljust(max_len), end=" ")
            else:
                print(f"{j}.{value}".ljust(max_len))
        if j % 5 != 0:
            print()
        choice = int(input(f"{i} (1 - {j}): "))
        x = unique_values_dict[i][choice-1]
    test_data.append(x)

test = pd.DataFrame([test_data], columns=features)

y_pred = model.predict(test)
print(f"Predicted Price: {y_pred}")
