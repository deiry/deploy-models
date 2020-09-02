FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install joblib scikit-learn

COPY ./model/model_neigh.joblib /model/
COPY ./app /app