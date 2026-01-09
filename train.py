import pandas as pd
import json, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create output directories
os.makedirs("output/model", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

# Load dataset
data = pd.read_csv("dataset/winequality-red.csv", sep=";")
X = data.drop("quality", axis=1)
y = data["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ✅ EXPERIMENT 8: Random Forest with tuned hyperparameters
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {
    "MSE": mse,
    "R2": r2
}

# Save outputs
joblib.dump(model, "output/model/model.pkl")

with open("output/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Print metrics (for logs)
print("Experiment 8 - Random Forest (n_estimators=200, max_depth=10)")
print("MSE:", mse)
print("R2:", r2)
