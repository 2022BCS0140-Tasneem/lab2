import pandas as pd
import json, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("output/model", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

data = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

joblib.dump(model, "output/model/model.pkl")

with open("output/results/metrics.json", "w") as f:
    json.dump({"MSE": mse, "R2": r2}, f, indent=4)

print("MSE:", mse)
print("R2:", r2)
