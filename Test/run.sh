#!/bin/bash
#SBATCH -J Test       
#SBATCH -p compute                           
#SBATCH -N 2  
#SBATCH --gres=gpu:titan_x_:1
#SBATCH --ntasks-per-node=1                               
#SBATCH --cpus-per-task=4  
#SBATCH -t 00:30:00  
#SBATCH -a 0-3                          
                            

input=(MNSOL Free TUM+Free ESOL)

source test.sh ${input[$SLURM_ARRAY_TASK_ID]}

