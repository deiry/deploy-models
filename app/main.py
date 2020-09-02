from fastapi import FastAPI
from joblib import load

from pydantic import  BaseModel, conlist
from typing import List

# define modidel for post request.
class Iris(BaseModel):
    data: List[conlist(float, min_items=4, max_items=4)]

clf = load('model_neigh.ml')

app = FastAPI(title="Iris ML API", description="API for iris dataset ml model", version="1.0")


def get_prediction(data: List):
    prediction = clf.predict(data).tolist()
    log_proba = clf.predict_proba(data).tolist()
    return {"prediction": prediction,
            "pred_proba": log_proba}
    
    
@app.post('/predict', tags=["predictions"])
async def predict(iris: Iris):
    data = dict(iris)['data']
    result = get_prediction(data)
    return result
    
