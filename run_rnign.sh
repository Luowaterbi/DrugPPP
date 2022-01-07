#!/bin/bash
#SBATCH -J RNIGN         
#SBATCH -p compute                           
#SBATCH -N 1   
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --ntasks-per-node=1                               
#SBATCH --cpus-per-task=4                     
#SBATCH -t 72:00:00    
#SBATCH -a 0-2                       
                            

source ~/.bashrc
conda activate pytorch

input=(atom mol_avg mol_sum)
source ./scripts/run_rnign_debug.sh 0 rnsa ${input[$SLURM_ARRAY_TASK_ID]}


