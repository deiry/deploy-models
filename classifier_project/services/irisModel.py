#Proporciona sugerencias de lo que recibe y debería devolver, en el código es ->
from typing import List

#Libreria para importar el modelo
import joblib

#Handlers de los logs
from loguru import logger

from classifier_project.models.prediction import PredictionResult

import numpy as np


class IrisModel(object):

    def __init__(self, path_model):
        self.path_model = path_model
        self.__load_local_model()
    
    def __load_local_model(self):
        self.model = joblib.load(self.path_model)

    def __pre_process(self,)