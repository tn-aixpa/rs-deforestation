
from hera.workflows import Workflow, DAG, Parameter
from digitalhub_runtime_hera.dsl import step

def pipeline():
    # Create a new Workflow with an entrypoint DAG and a parameter
    with Workflow(entrypoint="dag", arguments=[
        Parameter(name="geometry"),
        Parameter(name="outputName"),
        Parameter(name="startYear"),
        Parameter(name="endYear"),
        Parameter(name="shapeArtifactName"),
        Parameter(name="dataArtifactName")
        ]) as w:
        
        with DAG(name="dag"):
            string_dict_data = """{"satelliteParams":{"satelliteType":"Sentinel2",  "relativeOrbitNumber": "022"},"startDate":\""""+ str(w.get_parameter("startYear")) + """-01-01\","endDate": \"""" + str(w.get_parameter("endYear")) + """-12-31\","geometry": \"""" + str(w.get_parameter("geometry")) + """\","area_sampling":"true","cloudCover":"[0,5]","artifact_name": \"""" + str(w.get_parameter("dataArtifactName")) + """\"}"""          
             
            s1 = step(template={"action":"job",
                           "args":["main.py", string_dict_data], 
                           "secrets":["CDSETOOL_ESA_USER","CDSETOOL_ESA_PASSWORD"], 
                           "fs_group":"8877", 
                           "resources":{"mem": "32Gi", "cpu": "6"}, 
                           "envs":[{"name": "TMPDIR", "value": "/app/files"}], 
                           "volumes":[{"volume_type": "persistent_volume_claim","name": "volume-flood","mount_path": "/app/files","spec": { "size": "300Gi" }}]}, 
                 function="download_images_s2", 
                 name="download"
                 )
             
            s2 = step(template={"action":"job",
                           "args": ['/shared/launch.sh', str(w.get_parameter("shapeArtifactName")), str(w.get_parameter("dataArtifactName")), "[" + str(w.get_parameter("startYear")) + ',' + str(w.get_parameter("endYear")) + "]", str(w.get_parameter("outputName"))],
                           "fs_group":"8877",
                           "resources":{"cpu": "6","mem":"32Gi"},
                           "envs":[{"name": "TMPDIR", "value": "/app/data"}],
                           "volumes":[{"volume_type": "persistent_volume_claim", "name": "volume-deforestation","mount_path": "/app/data","spec": { "size": "250Gi" }}]},
                 function="elaborate",
                 name="elaborate"
                 )
            
            s1 >> s2
             
        return w
     
