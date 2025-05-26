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
echo "{'input1': '$1', 'input2': '$2', 'input3':$3, 'input4': '$4'}"
export PROJ_LIB=/home/nonroot/miniforge3/share/proj
export GDAL_DATA=/home/nonroot/miniforge3/share/gdal
echo "GDAL DATA AFTER EXPORT:"
echo $GDAL_DATA
echo "PROJ_LIB AFTER EXPORT"
echo $PROJ_LIB
python main.py "{'input1': '$1', 'input2': '$2', 'input3':$3, 'input4': '$4'}"
exit