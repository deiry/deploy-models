#Validacion de ingreso de datos
from pydantic import BaseModel

class PredictResult(BaseModel):
    iris_class: int
    predict_prob: float