# dash_app_with_mlflow



```bash

python3 -m venv .venv

# switch manually to virtual environment and then

$(.venv) python3 -m pip install --upgrade pip

$(.venv) pip install poetry

$(.venv) poetry install 
# will install all dependencies from the pyproject.toml file

```


We need to configure and save a `.env` file in the root folder of this Repo:

```bash

CONTAINER_NAME = "model-container"
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://localhost:10000/devstoreaccount1;QueueEndpoint=http://localhost:10001/devstoreaccount1"

MLFLOW_TRACKING_URI="http://localhost:5000"

MLFLOW_MODEL_DIRECTORY = "models:"
MLFLOW_MODEL_STAGE = "Staging"

```

because we have already some models in our local mlflow environment and we want to download them into our local folder.



## How to work with it.

Initially create the virtual environment

Create the .env file

Dump the models out of mlflow into the model folder in the dashapp folder, check the yaml file for the correct path.

play around
