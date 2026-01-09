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
    X, y, test_size=0.3, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = {
    "MSE": mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred)
}

joblib.dump(model, "output/model/model.pkl")
json.dump(metrics, open("output/results/metrics.json", "w"), indent=4)

print(metrics)
model = RandomForestRegressor(
    n_estimators=100,
    max_features="sqrt",
    random_state=42
)

