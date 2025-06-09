# How to monitor production data

The classifer model may be monitored in order to see the behavior of the model under production data.

## Exposing monitor gateway with Custom API

To deploy the monitoring gateway, it is possible to use the ``monitor`` operation defined in the project. Specifically, the following steps should be performed.

1. Register the ``monitor`` deployment operation in the project

```python
monitor = project.new_function(
    name="monitor", 
    kind="python", 
    python_version="PYTHON3_10", 
    code_src="monitor.py",     
    handler="serve",
    init_function="init",
    requirements=["SQLAlchemy==1.4.54", "psycopg2-binary"]
)
)
```
The function represent a Python Serverless function that should be deployed on cluster.

2. Activate the deployment.

The function aims at intercepting the calls to the classifier service and requires the following configuration provided:

- project secret ``DB_URL`` defined with the sql alchemy URL of the database where to store the monitored data.
- name of the table ``TABLE_NAME`` to write the data provided as environment variable when the service is deployed
- URL of the service ``SERVICE_URL`` to intercept as  environment variable when the service is deployed

```python
monitor_run = monitor.run(
    action="serve",
    envs=[
        {"name":"TABLE_NAME", "value": "faudit_classifier_monitor"}, 
        {"name": "SERVICE_URL", "value": service_url}],
    secrets=["DB_URL"]
)
```

Once the deployment is activated the monitor exposes the same API as a classifier and intercepts the requests.

3. Test the operation.

To test the functionality of the API behind the monitor, it is possible to use the V2 API calls. The "text" file contain the input text to be classified. The 'k' parameter specify the number of
classification labels required. For e.g. the request below asks for single classification label for input text.

```python
inputs = {"text": 'famiglia wifi ', "k": 1}
monitor_run.invoke(json={"inference_input": inputs}).text
```

Besides returning the output of the classification, the data of the call is written to the specified table in the database for further analysis.