#!/usr/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --partition=research
#SBATCH --account=zb7df-research-acct
#SBATCH --job-name=PACT-latentDCT
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --output=PACT-latentDCT-%j.txt

cd /home/zb7df/dev/PACT
conda init
conda activate pact
bash train_test.sh