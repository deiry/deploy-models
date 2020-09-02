# deploy-models
Welcome to the deploy-models wiki!

# Introducción

El código está disponible aqui. A continuación vamos a explorar acerca cómo desplegar fácilmente tus modelos en sklearn como un endpoint de API usando el framework fastAPI. Se recomienda estar familiarizado con sklearn y lo básico de APIs. 

## Antes de iniciar:

Aquí hay algunos recursos que pueden ayudarte a familiarizarte:


* Documentación de fastAPI [link](https://fastapi.tiangolo.com/)
* Introducción a APIs [link](https://www.freecodecamp.org/news/what-is-an-api-in-english-please-b880a3214a82/)

Ahora sí, para iniciar vamos a ejecutar los siguientes comandos.

`git clone https://github.com/deiry/deploy-models`

`cd deploy_models`

# Estructura del proyecto

En este tutorial se explicarán como desplegar un modelo de la forma sencilla y simple, sin embargo, si desean realizar una estructura más completa para un sistema más robusto pueden entrar aquí [link](https://github.com/eightBEC/fastapi-ml-skeleton/tree/master/fastapi_skeleton)

En este proyecto vamos a crear dos carpetas de la siguiente forma:

> model

> >  model_building.ipynb

> > model_neigh.joblib

> api
> > main.py

> Dockerfile


* model/model_building.ipynb -  Donde vamos a entrenar y guardar el modelo en un archivo con extensión 'joblib' 
* model/api_testing.ipynb - Donde vamos a probar la API, una vez se despliegue el modelo.
* model/model_1.joblib - Modelo guardado.
* app/main.py - ¡Aquí está la magia de la API!
* Dockerfile - Instancia de docker

