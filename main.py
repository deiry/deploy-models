from fastapi import FastAPI
from joblib import load
import sklearn
from pydantic import  BaseModel, conlist
from typing import List

# define model for post request.
class Iris(BaseModel):
    data: List[conlist(float, min_items=4, max_items=4)]

clf = load('neigh.ml')

app = FastAPI(title="Iris ML API", description="API for iris dataset ml model", version="1.0")


@app.post('/predict', tags=["predictions"])
async def get_prediction(iris: Iris):
    data = dict(iris)['data']
    prediction = clf.predict(data).tolist()
    log_proba = clf.predict_proba(data).tolist()
    return {"prediction": prediction,
            "pred_proba": log_proba}



