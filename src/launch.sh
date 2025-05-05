#!/bin/bash
ls -la /shared
cd ~
pwd
source .bashrc
echo "Init conda.."
source activate rsde
echo "py version:"
python -V

exit