# Elaboration

## 1. Register the `elaborate` operation in the project

```python
function_rs = proj.new_function(
    "elaborate",
    kind="container",
    image="ghcr.io/tn-aixpa/rs-deforestation:2.6_b8",
    command="/bin/bash",
    code_src="launch.sh"
    )
```

The function represent a container runtime that allows you to deploy deployments, jobs and services on Kubernetes. It uses the base image of rs-deforestation container deploved in the context of project create the environment required for the execution. It invovles pulling the base image with gdal installed and installing all the required libraries and launch instructions specified by 'launch.sh' file.

## 2. Run

The function aims at downloading all the deforestation inputs from projet context and perform the complex tax of deforesation elaboration.
