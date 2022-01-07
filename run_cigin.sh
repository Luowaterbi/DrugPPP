#!/bin/bash
#SBATCH -J CIGIN         
#SBATCH -p compute                           
#SBATCH -N 1   
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --ntasks-per-node=1                               
#SBATCH --cpus-per-task=4                     
#SBATCH -t 120:00:00                            
                            

source ~/.bashrc
conda activate pytorch

source ./scripts/run_cigin_test.sh 0