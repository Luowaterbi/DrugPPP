#!/bin/bash
#SBATCH -J RNIGN         
#SBATCH -p compute                           
#SBATCH -N 1   
#SBATCH --gres=gpu:titan_v:1
#SBATCH --ntasks-per-node=1                               
#SBATCH --cpus-per-task=4                     
#SBATCH -t 120:00:00                          
                            

source ~/.bashrc
conda activate pytorch

# input=(atom mol_avg mol_sum)
# wandb agent luowaterbi/DrugPP/teui8hqc &
source ./scripts/run_rnign1.sh 0
# sleep 259200
# sleep 432000


