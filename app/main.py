from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/health")
def health():
    return {
        "name": "Sravan",
        "roll_no": "2022BCS0023"
    }

@app.post("/predict")
def predict(data: dict):
    prediction = model.predict([list(data.values())])[0]

    return {
        "prediction": int(prediction),
        "name": "Sravan",
        "roll_no": "2022BCS0023"
    }