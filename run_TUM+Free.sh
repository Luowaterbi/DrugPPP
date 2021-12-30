#!/bin/bash
#SBATCH -J Add_TUM+Free          
#SBATCH -p compute                           
#SBATCH -N 1   
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --ntasks-per-node=1                               
#SBATCH --cpus-per-task=4                     
#SBATCH -t 120:00:00                            
                            

source ~/.bashrc
conda activate pytorch

source ./scripts/run_rnign_TUM+Free.sh 0

