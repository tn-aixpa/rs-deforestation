
# Workflow

<p align="justify">In this step we will create a workflow pipeline that establish a clear, repeatable process for handling the set of scenario tasks (download, elaborate). The DH platform pipeline ensures that tasks are completed in a sepcific order. It also provide the ease to fine tune the steps as per requirements of scenario imporving efficiency, consistency, aand traceability. For more detailed information about workflow and their management see the <a href="https://scc-digitalhub.github.io/docs/tasks/workflows">documentation</a>. Inside the project 'src' folder there exist a jypter notebook <a href="../../src/workflow.ipynb">workflow.ipynb</a> that depicts the creation and management of workflow.</p>

## 1. Initialize the project

<p align="justify">Create the working context: data management project for scenario. Project is a placeholder for the code, data, and management of the data operations and workflows. To keep it reproducible, we use the git source type to store the definition and code.</p>

```python
import digitalhub as dh
PROJECT_NAME = "deforestation" # here goes the project name that you are creating on the platform
proj = dh.get_or_create_project(PROJECT_NAME)
```

## 2. Log shape artifact

<p align="justify">Log the shape file 'bosco' which can be downloaded from the <a href="https://webgis.provincia.tn.it/">WebGIS Portal</a> confine del bosco layer or using direct <a href="https://siatservices.provincia.tn.it/idt/vector/p_TN_3d0874bc-7b9e-4c95-b885-0f7c610b08fa.zip">link</a> to zip file. Unzip the files in a folder named 'bosco' and then log it</p>

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

## 3. Register 'Download' operation in the project

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

<p align="justify">Register 'download_images_s2' operation in the project. The function if of kind container runtime that allows you to deploy deployments, jobs and services on Kubernetes. It uses the base image of sentinel-tools deploved in the context of project which is a wrapper for the Sentinel download and preprocessing routine for the integration with the AIxPA platform. For more details click <a href="https://github.com/tn-aixpa/sentinel-tools/)">here</a></p>

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
    image="ghcr.io/tn-aixpa/rs-deforestation:2.7_b4",
    command="/bin/bash",
    code_src="launch.sh"
    )
```

<p align="justify">The function represent a container runtime that allows you to deploy deployments, jobs and services on Kubernetes. It uses the base image of rs-deforestation container deploved in the context of project create the environment required for the execution. It invovles pulling the base image with gdal installed and installing all the required libraries and launch instructions specified by 'launch.sh' file.</p>

## 5. Create workflow pipeline

<p align="justify">Workflows can be created and managed as entities similar to functions. From the console UI one can access them from the dashboard or the left menu. Run the following step to create 'workflow' python source file inside src directory. The workflow handler takes as input</p>

- startYear (start year for time series analysis)
- endYear (end year from time series analysis)
- geometry (area of interest)
- shapeArtifactName (shape forest mask artifact name already registered in step2)
- dataArtifactName (optional)
- outputName (output artifact name)

<p align="justify">The inputs are used inside to the workflow among different functions. The first step performs sentinel downloads using the function created in previous step. The download function takes as input a list of arguments (args=["main.py", string_dict_data]) where the first argument is the python script file that will be launched inside to the container and the second argument is the json input string which includes all the necessary parameters of sentinel download operation like date, geometry, product type, cloud cover etc. For more details click <a href="https://github.com/tn-aixpa/sentinel-tools/">here</a>. The last step of workflow perform elaboration using the 'elaborate' function created in previous step. The elaboration function taks as input a list of arguments where the first argument is the bash script that will be launched on entry inside to the container while the following parameters contains both fixed and dynamic parameters. The fixed parameter includes project artifacts names (data_s2_v2), which is downloaded as the result of first step. The set of dynamic parameters included outputName, startYear, endYear, geometry etc. which can be passed as input to the main workflow. The workflow can be adopted as per context needs by modifying/passing the different parametric values as depicted in 'Register workflow' section.</p>

```python
%%writefile "deforestation_pipeline.py"

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

There is a committed version of this file on the repo.

## 6. Register workflow

<p align="justify">Register workflow 'pipeline_deforestation' in the project. In the following step, we register the workflow using the committed version of pipeline source code on project git repository. It is required to update the 'code_src' url with github username and personal access token in the code cell below</p>

```python
workflow = proj.new_workflow(
name="pipeline_deforestation",
kind="kfp",
code_src="git+https://<username>:<personal_access_token>@github.com/tn-aixpa/rs-deforestation",
handler="src.deforestation_pipeline:myhandler")
```

<p align="justify">If you want to modify the pipeline source code, either update the existing version on github repo or register the pipeline with local version of python source file generated in prevous step for e.g. the value of parameter 'dataArtifactName' is optional and set to 'data_s2_v2' in committed version on project repo. If you want to log it with different name inside to the DH platform project, create/update the pipeline code locally by replacing the string with parameter followed by the registration as shown below.</p>

```python
workflow = proj.new_workflow(name="pipeline_deforestation", kind="kfp", code_src= "deforestation_pipeline.py", handler = "myhandler")
```

## 7. Build workflow

```python
wfbuild = workflow.run(action="build", wait=True)
wfbuild.spec
```

After the build, the pipeline specification and configuration is displayed as the result of this step(wfbuild.spec). The same can be achieved from the console UI dashboard or the left menu using the 'INSPECTOR' button which opens a dialog containing the resource in JSON format.

```python
{
    'task': 'kfp+pipeline://deforestation/ed9a9f8941af4cdab2715048b0902dc3',
    'local_execution': False,
    'workflow': 'kfp://deforestation/pipeline_deforestation:32c08ce786d041ae8617089ee390e91a',
    ...
  }
```

<p align="justify">In order to integrate the pipeline with the front end UI 'rsde-pipeline-manger', the value of 'task' and 'workflow' keys are the two important configuration parameters that must be set in the in the configuration(config.yml) as taskId and workflowId. For more detailed information see [rsde-pipeline-manger](https://github.com/tn-aixpa/rsde-pipeline-manager)</p>
