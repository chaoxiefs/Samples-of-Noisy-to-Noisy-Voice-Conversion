#!/bin/bash
#SBATCH --output=
#SBATCH --error=
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time 10-00:00:00

source */tools/venv/bin/activate
python preprocess.py
python generate_scp.py
python train.py
python cvloop-loop.py
python cal_mcd_loop.py
python cal_sim_loop.py