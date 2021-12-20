#!/bin/bash
#SBATCH -J MOE_in_RNIGN          
#SBATCH -p compute              
#SBATCH -o ./slurm/MOE_in_RNIGN.out              
#SBATCH -N 1   
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --ntasks-per-node=1                               
#SBATCH --cpus-per-task=4                     
#SBATCH -t 72:00:00                            
                            

source ~/.bashrc
conda activate pytorch

source ./scripts/run_rnign.sh 0