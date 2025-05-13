# Easy model deployment with FastAPI

In this short tutorial we are going to learn how to deploy our sklearn models with FastAPI in a simple way. It is expected that you are already familiar with sklearn, APIs and fastAPI.

If you want to learn more, here are some good links to start with:

- https://fastapi.tiangolo.com/
- https://www.freecodecamp.org/news/what-is-an-api-in-english-please-b880a3214a82/

We are going to use Docker in this tutorial, please install it if you don't have it already.

---

# Structure

We are going to show a very basic structure. If you want something more production ready you can check this example:
https://github.com/eightBEC/fastapi-ml-skeleton/tree/master/fastapi_skeleton

Your structure will look like this:

```
project/
│
├── model/
│   ├── model_building.ipynb
│   ├── api_testing.ipynb
│   └── model_neigh.joblib
│
├── app/
│   └── main.py
│
└── Dockerfile
```

### model_building.ipynb

Notebook used to build and save your model

### api_testing.ipynb

Notebook to test your API

### model_neigh.joblib

The trained model

### main.py

Your fastAPI code

### Dockerfile

Code to generate your image with all your dependencies

---

# Build your model

You can use any of the notebooks in this course: https://github.com/jdariasl/ML_IntroductoryCourse/tree/master/Labs

In this case we are going to use the iris dataset and train a KNN model.

```python
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
```

Split the dataset into training and test datasets with a 70%-30% proportion

```python
(X, y) = datasets.load_iris(return_X_y=True)

N = np.size(X, 0)
split = int(N * 0.7)

X_train = X[0:split]
y_train = y[0:split]

X_test = X[split:N]
y_test = y[split:N]
```

Train the model

```python
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
```

Save the model

```python
from joblib import dump
dump(neigh, 'model_neigh.joblib')
```

Save the test data to use it later

```python
np.savetxt('X_test.csv', X_test, delimiter=',')
```

---

# Creating the API

Move to the /app directory and create the main.py file

```python
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, conlist
from typing import List
```

Define the format for the input data

```python
class Iris(BaseModel):
    data: List[conlist(float, min_items=4, max_items=4)]
```

This means a list of elements with length 4.

Load the model and create the API app

```python
clf = load('model_neigh.joblib')

app = FastAPI(
    title="Iris ML API",
    description="API for iris dataset ml model",
    version="1.0"
)
```

Create the prediction function

```python
def get_prediction(data: List):
    prediction = clf.predict(data).tolist()
    log_proba = clf.predict_proba(data).tolist()

    return {
        "prediction": prediction,
        "pred_proba": log_proba
    }
```

The result of the model is a numpy array. We use tolist to convert it into a python list.

Create the endpoint

```python
@app.post('/predict', tags=["predictions"])
async def predict(iris: Iris):
    data = dict(iris)['data']
    result = get_prediction(data)
    return result
```

---

# Dockerfile

```Dockerfile
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install joblib scikit-learn

COPY ./model/ /model/
COPY ./app /app
```

## Create the image

```bash
docker build -t myapi .
```

## Run the container

```bash
docker run -d --name myapicontainer -p 80:80 myapi
```

Now your API should be running on

```
http://localhost/predict
```

---

# Test the API

We go to api_testing.ipynb and create the test

```python
import requests
import numpy as np
import pandas as pd
```

```python
df = pd.read_csv('/model/X_test.csv')
X_test = df.to_numpy()
data = {"data": X_test.tolist()}
r = requests.post('http://localhost/predict', json=data)
r.json()
```

```json
{'prediction': [2, 1, 2, 2, 2, 1, 1, 2, 1, 1],
 'pred_proba': [[0.0, 0.0, 1.0],
  [0.0, 1.0, 0.0],
  [0.0, 0.0, 1.0],
  [0.0, 0.0, 1.0],
  [0.0, 0.0, 1.0],
  [0.0, 1.0, 0.0],
  [0.0, 1.0, 0.0],
  [0.0, 0.0, 1.0],
  [0.0, 1.0, 0.0],
  [0.0, 1.0, 0.0]]}
```

And that’s it!
