from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Initialize the FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return RedirectResponse(url="/docs#/default/predict_predict_post")

# Define the request body using Pydantic
class PredictionRequest(BaseModel):
    Temperature_C: float
    Humidity: float
    eCO2_ppm: float

# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert the request data to a DataFrame
    data = {
        'Temperature[C]': [request.Temperature_C],
        'Humidity[%]': [request.Humidity],
        'eCO2[ppm]': [request.eCO2_ppm]
    }
    df = pd.DataFrame(data)
    # Make predictions
    prediction = model.predict(df)
    # Return the prediction as JSON
    return {'prediction': int(prediction[0])}

# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
