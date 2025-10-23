
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
                     envs=[{"name": "TMPDIR", "value": "/app/files"}],
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
                        "mount_path": "/app/data",
                        "spec": { "size": "250Gi" }
                    }],
                     args=['/shared/launch.sh', str(shapeArtifactName), 'data_s2_v2', "[" + str(startYear) + ',' + str(endYear) + "]", str(outputName)]
                     ).after(s1)
     
