#!/bin/bash
#SBATCH -J MoE_in_Free          
#SBATCH -p compute                           
#SBATCH -N 1   
#SBATCH --gres=gpu:titan_v:1
#SBATCH --ntasks-per-node=1                               
#SBATCH --cpus-per-task=4                     
#SBATCH -t 120:00:00                            
                            

source ~/.bashrc
conda activate pytorch

source ./scripts/run_rnign_Free.sh 0

