#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --mem-per-cpu=1.5G #increase if needed
#SBATCH --time=1:00:00

module load python/3.7.0
virtualenv --no-download ~/py37
source ~/py37/bin/activate
pip install --upgrade pip 
pip install --no-index -r requirements.txt
python filename.py


