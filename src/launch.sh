#!/bin/bash
ls -la /shared
cd ~
pwd
source .bashrc
echo "Init conda.."
source activate rsde
echo "GDAL version:"
python --version
gdal-config --version
#printenv
cd /app
python main.py "{'input1':'bosco', 'input2': 'data', 'input3':['2018', '2019'], 'input4': 'deforestation_output'}"
exit