How to prepare data for elaboration

To prepare the deforestation data, it is required to log the data in the project context

1. Initialize the project

```python
import digitalhub as dh
PROJECT_NAME = "deforestation" # here goes the project name that you are creating on the platform
project = dh.get_or_create_project(PROJECT_NAME)
```

2. Log the artifact

```python
artifact = project.log_artifact(name="train_data_it",
                    kind="artifact",
                    source="./addestramento.gzip")
```

Note that to invoke the operation on the platform, the data should be avaialble as an artifact on the platform datalake.

```python
artifact = project.get_artifact("train_data_it")
artifact.key
```

The resulting dataset will be registered as the project artifact in the datalake under the name `train_data_it`.

Register to the open data space copenicus(if not already) and get your credentials.

https://identity.dataspace.copernicus.eu/auth/realms/CDSE/login-actions/registration?client_id=cdse-public&tab_id=FIiRPJeoiX4

Log the credentials as project secret keys as shown below

# How to expose classifier model

The classifer model may be exposed as an API for classification. In this project we have used the Custom API approach to serve the generated model.

## Exposing model with Custom API

To deploy more specific API that takes into account the types of labels, it is possible to use the `serve` operation defined in the project. Specifically, the following steps should be performed.

1. Register the `serve` deployment operation in the project

```python
func = project.new_function(
    name="serve",
    kind="python",
    python_version="PYTHON3_10",
    code_src="git+https://<username>:<personal_access_token>@github.com/tn-aixpa/faudit-classifier",
    handler="src.serve:serve",
    init_function="init",
    requirements=["numpy<2", "pandas==2.1.4","transformer_engine==1.12.0", "transformer_engine_cu12==1.12.0", "transformers==4.46.3", "torch==2.5.1", "torchmetrics==1.6.0"]
)
```

The function represent a Python Serverless function that should be deployed on cluster.

2. Activate the deployment.

```python
serve_run = func.run(
    action="serve"
)
```

Once the deployment is activated, the V2 Open Inference Protocol is exposed and the Open API specification is available under `/docs` path.

3. Test the operation.

To test the functionality of the API, it is possible to use the V2 API calls. The "text" file contain the input text to be classified. The 'k' parameter specify the number of
classification labels required. For e.g. the request below asks for single classification label for input text.

```python
inputs = {"text": 'famiglia wifi ', "k": 1}
serve_run.invoke(json={"inference_input": inputs}).text
```

The api response will return the ids of most probable taxonomy. For futher details, look in to the correspondence.csv file present inside src folder which provide mapping between the ids and related taxonomy.

```
{
    "results": [
        46
    ]
}
```
