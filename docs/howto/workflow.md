# Workflow

In this step we will create a workflow pipeline that establish a clear, repeatable process for handling the set of scenario tasks (download, elaborate). The DH platform pipeline ensures that tasks are completed in a sepcific order. It also provide the ease to fine tune the steps as per requirements of scenario imporving efficiency, consistency, aand traceability. For more detailed information about workflow and their management see the [documentation](https://scc-digitalhub.github.io/docs/tasks/workflows). Insie the project 'src' folder there exist a jypter notebook(workflow.ipynb) that depicts the creation and management of workflow.

## 1. Initialize the project

Create the working context: data management project for scenario. Project is a placeholder for the code, data, and management of the data operations and workflows. To keep it reproducible, we use the git source type to store the definition and code.

```python
import digitalhub as dh
PROJECT_NAME = "deforestation" # here goes the project name that you are creating on the platform
proj = dh.get_or_create_project(PROJECT_NAME)
```

We convert the data management ETL operations into functions - single executable operations that can be executed in the platform. Create a directory named 'src' to save all the python source files.

```python
import os
directory="src"
if not os.path.exists(directory):
    os.makedirs(directory)
```

## 2. Log the Shape artifact

Log the shape file 'bosco' which can be downloaded from the [WebGIS Portal](https://webgis.provincia.tn.it/) confine del bosco layer or from https://siatservices.provincia.tn.it/idt/vector/p_TN_3d0874bc-7b9e-4c95-b885-0f7c610b08fa.zip. Unzip the files in a folder named 'bosco' and then log it

```python
artifact_name='bosco'
src_path='bosco'
artifact_bosco = proj.log_artifact(name=artifact_name, kind="artifact", source=src_path)
```

Note that to invoke the operation on the platform, the data should be avaialble as an artifact on the platform datalake.

```python
artifact = proj.get_artifact("bosco")
artifact.key
```

The resulting dataset will be registered as the project artifact in the datalake under the name `bosco`.

## 3. register 'Download' operation in the project

Register to the open data space copernicus(if not already) and get your credentials.

```
https://identity.dataspace.copernicus.eu/auth/realms/CDSE/login-actions/registration?client_id=cdse-public&tab_id=FIiRPJeoiX4
```

Log the credentials as project secret keys as shown below

```python
# THIS NEED TO BE EXECUTED JUST ONCE
secret0 = proj.new_secret(name="CDSETOOL_ESA_USER", secret_value="esa_username")
secret1 = proj.new_secret(name="CDSETOOL_ESA_PASSWORD", secret_value="esa_password")
```

```python
string_dict_data = """{
 "satelliteParams":{
     "satelliteType": "Sentinel2"
 },
 "startDate": "2018-01-01",
 "endDate": "2019-12-31",
 "geometry": "POLYGON((10.968432350469937 46.093829019481056,10.968432350469937 46.09650743619973, 10.97504139531014 46.09650743619973,10.97504139531014 46.093829019481056, 10.968432350469937 46.093829019481056))",
 "area_sampling": "true",
 "cloudCover": "[0,5]",
 "artifact_name": "data_s2_deforestation"
 }"""

list_args =  ["main.py",string_dict_data]
```

Register 'download_images_s2' operation in the project. The function if of kind container runtime that allows you to deploy deployments, jobs and services on Kubernetes. It uses the base image of sentinel-tools deploved in the context of project which is a wrapper for the Sentinel download and preprocessing routine for the integration with the AIxPA platform. For more details [Click here](https://github.com/tn-aixpa/sentinel-tools/). The parameters passed for sentinel downloads includes the starts and ends dates corresponding to period of two years of data. The ouput of this step will be logged inside to the platfrom project context as indicated by parameter 'artifact_name' ('data_s2_deforestation').Several other paramters can be configures as per requirements for e.g. geometry, cloud cover percentage etc.

```python
function_s2 = proj.new_function(
    "download_images_s2",
    kind="container",
    image="ghcr.io/tn-aixpa/sentinel-tools:0.11.5",
    command="python")
```

## 4. Register the `elaborate` operation in the project

```python
function_rs = proj.new_function(
    "elaborate",
    kind="container",
    image="ghcr.io/tn-aixpa/rs-deforestation:2.7_b2",
    command="/bin/bash",
    code_src="launch.sh"
    )
```

The function represent a container runtime that allows you to deploy deployments, jobs and services on Kubernetes. It uses the base image of rs-deforestation container deploved in the context of project create the environment required for the execution. It invovles pulling the base image with gdal installed and installing all the required libraries and launch instructions specified by 'launch.sh' file.

## 5. Create workflow pipeline

Workflows can be created and managed as entities similar to functions. From the console UI one can access them from the dashboard or the left menu.
run the following step to create 'workflow' python source file inside src directory. The workflow handler takes as input

- startYear (start year for time series analysis)
- endYear (end year from time series analysis)
- geometry (area of interest)
- shapeArtifactName (shape forest mask artifact name already register in step2)
- dataArtifactName (optional)
- outputName (output artifact name)

```python
%%writefile "src/deforestation_pipeline.py"

from digitalhub_runtime_kfp.dsl import pipeline_context

def myhandler(startYear, endYear, geometry, shapeArtifactName, dataArtifactName, outputName):
    with pipeline_context() as pc:
        string_dict_data = """{"satelliteParams":{"satelliteType":"Sentinel2"},"startDate":\""""+ str(startYear) + """-01-01\","endDate": \"""" + str(endYear) + """-12-31\","geometry": \"""" + str(geometry) + """\","area_sampling":"true","cloudCover":"[0,5]","artifact_name":"data_s2_v2"}"""
        s1 = pc.step(name="download",
                     function="download_images_s2",
                     action="job",
                     secrets=["CDSETOOL_ESA_USER","CDSETOOL_ESA_PASSWORD"],
                     fs_group='8877',
                     args=["main.py", string_dict_data],
                     volumes=[{
                        "volume_type": "persistent_volume_claim",
                        "name": "volume-deforestation",
                        "mount_path": "/app/files",
                        "spec": { "size": "250Gi" }
                        }
                    ])
        s2 = pc.step(name="elaborate",
                     function="elaborate",
                     action="job",
                     fs_group='8877',
                     resources={"cpu": {"requests": "6", "limits": "12"},"mem":{"requests": "32Gi", "limits": "64Gi"}},
                     volumes=[{
                        "volume_type": "persistent_volume_claim",
                        "name": "volume-deforestation",
                        "mount_path": "/app/files",
                        "spec": { "size": "250Gi" }
                    }],
                     args=['/shared/launch.sh', str(shapeArtifactName), 'data_s2_v2', "[" + str(startYear) + ',' + str(endYear) + "]", str(outputName)]
                     ).after(s1)
```

## 6. Register workflow

Register workflow 'pipeline_deforestation' in the project.

```python
workflow = proj.new_workflow(name="src/pipeline_deforestation", kind="kfp", code_src= "deforestation_pipeline.py", handler = "myhandler")
```

## 7. Build workflow

```python
wfbuild = workflow.run(action="build", wait=True)
wfbuild.spec
```

After the build, the pipeline specification and configuration is displayed as the result of this step(wfbuild.spec). The same can be achieved from the console UI dashboard or the left menu.

```python
{
    'task': 'kfp+pipeline://deforestation/ed9a9f8941af4cdab2715048b0902dc3',
    'local_execution': False,
    'workflow': 'kfp://deforestation/pipeline_deforestation:32c08ce786d041ae8617089ee390e91a',
    ...
  }
```

In order to integrate the pipeline with the front end UI 'rsde-pipeline-manger', the value of 'task' and 'workflow' keys are the two important configuration parameters that must be set in the in the configuration(config.yml) as taskId and workflowId. For more detailed information see [rsde-pipeline-manger](https://github.com/tn-aixpa/rsde-pipeline-manager)
