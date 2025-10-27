
import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)


class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

class PredictResponse(BaseModel):
    conversion_probability : float
    will_convert : bool

app = FastAPI(title = 'Prediction for Conversion')

@app.post("/predict", response_model = PredictResponse)
def predict(client: Client):
    record = client.dict()
    prob = pipeline.predict_proba([record])[0, 1]
    will_convert = prob >= 0.5

    return PredictResponse(
        conversion_probability = float(prob),
        will_convert = will_convert
    )

if __name__ == '__main__':
    uvicorn.run(app, host = '0.0.0.0', port = 3000)

