import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel


np.random.seed(0)
n = 2000

temperature = np.random.normal(70, 10, n)
vibration = np.random.normal(5, 2, n)
pressure = np.random.normal(100, 15, n)
runtime_hours = np.random.uniform(0, 5000, n)

# failure becomes more likely with high temp, high vibration, long runtime
failure_score = (
    0.03 * (temperature - 70) +
    0.5 * (vibration - 5) +
    0.0006 * runtime_hours +
   np.random.normal(0, 1, n)
)
failure = (failure_score > 2).astype(int)

df = pd.DataFrame({
    "Temperature": temperature,
    "Vibration": vibration,
    "Pressure": pressure,
    "Runtime Hours": runtime_hours,
    "Failure": failure
})

print(df.head())
print(f"\nFailure rate: {df['Failure'].mean():.2%}")

X = df[["Temperature", "Vibration", "Pressure", "Runtime Hours"]]
y = df["Failure"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = XGBClassifier(random_state=0)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print("\n", classification_report(y_test, predictions))

# ratio of negative to positive examples in training data
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
weight = neg / pos
print(f"scale_pos_weight: {weight:.2f}")

model_weighted = XGBClassifier(random_state=0, scale_pos_weight=weight)
model_weighted.fit(X_train, y_train)

predictions_weighted = model_weighted.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, predictions_weighted):.4f}")
print("\n", classification_report(y_test, predictions_weighted))

mlflow.set_experiment("predictive_maintenance")

with mlflow.start_run(run_name="baseline_xgboost"):
    mlflow.log_param("scale_pos_weight", 1.0)
    mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))

with mlflow.start_run(run_name="weighted_xgboost"):
    mlflow.log_param("scale_pos_weight", weight)
    mlflow.log_metric("accuracy", accuracy_score(y_test, predictions_weighted))

model_weighted.save_model("failure_model.json")
print("Model Saved!")

app = FastAPI()

model = XGBClassifier()
model.load_model("failure_model.json")

@app.get("/")
def home():
    return {"message": "Predictive maintenance API is running"}

class SensorReading(BaseModel):
    temperature: float
    vibration: float
    pressure: float
    runtime_hours: float

@app.post("/predict")
def predict(reading: SensorReading):
    X = np.array([[reading.temperature, reading.vibration, reading.pressure, reading.runtime_hours]])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    return {
        "failure_predicted": bool(prediction),
        "failure_probability": float(probability)
    }