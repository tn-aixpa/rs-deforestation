#!/bin/bash
ls -la /shared
cd ~
pwd
source .bashrc
echo "Init conda.."
source activate rsde
echo "GDAL version:"
gdal-config --version
printenv | grep DH
exit