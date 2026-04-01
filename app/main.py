from fastapi import FastAPI
import joblib

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

@app.get("/health")
def health():
    return {
        "name": "Sravan",
        "roll_no": "2022BCS0023"
    }
@app.post("/predict")
def predict(data: dict):
    try:
        values = list(data.values())
        prediction = model.predict([values])[0]

        return {
            "prediction": int(prediction),
            "name": "Sravan",
            "roll_no": "2022BCS0023"
        }
    except Exception as e:
        return {"error": str(e)}