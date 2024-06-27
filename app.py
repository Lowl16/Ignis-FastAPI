from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Define the request body using Pydantic
class PredictionRequest(BaseModel):
    Temperature_C: float
    Humidity: float
    eCO2_ppm: float

# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert the request data to a numpy array
    data = np.array([[request.Temperature_C, request.Humidity, request.eCO2_ppm]])
    
    # Make predictions
    prediction = model.predict(data)
    
    # Return the prediction as JSON
    return {'prediction': int(prediction[0])}

# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
