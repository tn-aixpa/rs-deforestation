#!/bin/bash
ls -la /shared
cd ~
pwd
source .bashrc
echo "GDAL version:"
gdal-config --version
python --version
#printenv
cd /app
echo "{'input1': '$1', 'input2': '$2', 'input3':$3, 'input4': '$4'}"
#python main.py "{'input1':'bosco', 'input2': 'data', 'input3':['2018', '2019'], 'input4': 'deforestation_output'}"
python main.py "{'input1': '$1', 'input2': '$2', 'input3':$3, 'input4': '$4'}"
exit