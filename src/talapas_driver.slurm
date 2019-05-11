#!/bin/bash
#SBATCH --job-name=Stratego              # Job Name
##SBATCH --partition=gpu                   # Short GPU jobs (<1 days)
#SBATCH --partition=longgpu               # Long GPU jobs (1-14 days)
#SBATCH --time=1-00:00:00                 # Time limit (dd-hh:mm:ss)
#SBATCH --mem-per-cpu=32G                 # Amount of RAM required per node
#SBATCH --nodes=1                         # Number of nodes to use
#SBATCH --cpus-per-task=4                 # Number of cores for each task
#SBATCH --ntasks-per-node=1               # Number of tasks to launch per node
#SBATCH --gres=gpu:1                      # General REServation of gpu:number of gpus
#SBATCH --mail-type=ALL                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zhammoud@uoregon.edu  # Where to send progress related emails

# Load Libraries and modules needed by the script
module load cuda/9.0
module load python3

python3 driver.py
