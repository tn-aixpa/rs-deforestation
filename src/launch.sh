#!/bin/bash
ls -la /shared
cd ~
pwd
source .bashrc
echo "GDAL version:"
gdal-config --version
python --version
echo "GDAL DATA:"
echo $GDAL_DATA
echo "PROJ_LIB"
echo $PROJ_LIB
cd /app
echo "{'shapeArtifactName': '$1', 'dataArtifactName': '$2', 'years':$3, 'outputArtifactName': '$4'}"
export PROJ_LIB=/home/nonroot/miniforge3/share/proj
export GDAL_DATA=/home/nonroot/miniforge3/share/gdal
#export PATH="/home/nonroot/miniforge3/snap/.snap/auxdata/gdal/gdal-3-0-0/bin/:$PATH"
echo "GDAL DATA AFTER EXPORT:"
echo $GDAL_DATA
echo "PROJ_LIB AFTER EXPORT"
echo $PROJ_LIB
python main.py "{'shapeArtifactName': '$1', 'dataArtifactName': '$2', 'years':$3, 'outputArtifactName': '$4'}"
exit