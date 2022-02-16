#!/bin/bash
#SBATCH -J RNIGN         
#SBATCH -p compute                           
#SBATCH -N 1   
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --ntasks-per-node=1                               
#SBATCH --cpus-per-task=4                     
#SBATCH -t 120:00:00                          
                            

source ~/.bashrc
conda activate pytorch


python main.py --batch_size=26 --cross_dst=500000 --d_model=64 --data_dir=./data/MNSOL --enc_dropout=0.04642068980268356 --enc_n_head=2 --enc_n_layer=6 --inter_dropout=0.19509414221674437 --inter_n_head=2 --inter_n_layer=4 --lambda_attention=0.3675638214915888 --lambda_distance=0.1811487470649024 --lr=0.0003284251602409876 --max_epochs=277 --mix=1 --moe=1 --moe_dropout=0.2917086861805982 --moe_input=atom --moe_loss_coef=0.013194671302851985 --name=rnign.8qqtg4xi --num_experts=95 --num_used_experts=5 --output_dir=./runs/MNSOL/rnign.0122.sweep.moe/MNSOL --patience=17 --readout=rn_avg --rn_dst=2 --task=sol --warmup_proportion=0.0890068073831991 --weight_decay=0.19762006242575583 >./log/MNSOL/for_moe_loss/8qqtg4xi.log &
python main.py --batch_size=60 --cross_dst=1000000 --d_model=64 --data_dir=./data/MNSOL --enc_dropout=0.18713691760865975 --enc_n_head=8 --enc_n_layer=4 --inter_dropout=0.2464418758838493 --inter_n_head=4 --inter_n_layer=2 --lambda_attention=0.48464849628749174 --lambda_distance=0.23797964583787148 --lr=0.0007222714934525658 --max_epochs=243 --mix=1 --moe=1 --moe_dropout=0.1509553954299643 --moe_input=atom --moe_loss_coef=0.01494155287541929 --name=rnign.w2dmablg --num_experts=40 --num_used_experts=6 --output_dir=./runs/MNSOL/rnign.0122.sweep.moe/MNSOL --patience=19 --readout=rn_sum --rn_dst=2 --task=sol --warmup_proportion=0.07761718390930615 --weight_decay=0.10412708896503184 >./log/MNSOL/for_moe_loss/w2dmablg.log &
python main.py --batch_size=53 --cross_dst=1500000 --d_model=64 --data_dir=./data/MNSOL --enc_dropout=0.038435472358963024 --enc_n_head=2 --enc_n_layer=8 --inter_dropout=0.3338757729458822 --inter_n_head=2 --inter_n_layer=5 --lambda_attention=0.3839821893352833 --lambda_distance=0.2686257025935189 --lr=0.0008431159942366702 --max_epochs=430 --mix=1 --moe=1 --moe_dropout=0.3150281641826589 --moe_input=mol_avg --moe_loss_coef=0.01701528041634721 --name=rnign.l4m1dle4 --num_experts=100 --num_used_experts=1 --output_dir=./runs/MNSOL/rnign.0122.sweep.moe/MNSOL --patience=16 --readout=rn_avg --rn_dst=2 --task=sol --warmup_proportion=0.04869716802119739 --weight_decay=0.17301228851239622 >./log/MNSOL/for_moe_loss/l4m1dle4.log &
python main.py --batch_size=68 --cross_dst=1000000 --d_model=64 --data_dir=./data/MNSOL --enc_dropout=0.046367509561734965 --enc_n_head=4 --enc_n_layer=7 --inter_dropout=0.3539850331940397 --inter_n_head=8 --inter_n_layer=3 --lambda_attention=0.5593676705487147 --lambda_distance=0.4106189397578005 --lr=0.0009600393662413222 --max_epochs=321 --mix=1 --moe=1 --moe_dropout=0.2294759067096032 --moe_input=atom --moe_loss_coef=0.008364937975676476 --name=rnign.j5631hwv --num_experts=94 --num_used_experts=2 --output_dir=./runs/MNSOL/rnign.0122.sweep.moe/MNSOL --patience=18 --readout=rn_avg --rn_dst=2 --task=sol --warmup_proportion=0.06099105952970614 --weight_decay=0.19236374478700763 >./log/MNSOL/for_moe_loss/j5631hwv.log






# input=(atom mol_avg mol_sum)
# source ./scripts/run_rnign_debug.sh 0
# sleep 259200
# sleep 432000
# wandb agent luowaterbi/DrugPP/bb3phmgf &
# wandb agent luowaterbi/DrugPP/bb3phmgf &
# wandb agent luowaterbi/DrugPP/bb3phmgf &
# wandb agent luowaterbi/DrugPP/bb3phmgf &
# wandb agent luowaterbi/DrugPP/bb3phmgf &
# wandb agent luowaterbi/DrugPP/bb3phmgf &
# wandb agent luowaterbi/DrugPP/bb3phmgf &
# wandb agent luowaterbi/DrugPP/bb3phmgf &
# wandb agent luowaterbi/DrugPP/bb3phmgf &
# wandb agent luowaterbi/DrugPP/bb3phmgf 
# wandb agent luowaterbi/DrugPP/5jc5rieb &
# wandb agent luowaterbi/DrugPP/5jc5rieb &
# wandb agent luowaterbi/DrugPP/5jc5rieb &
# wandb agent luowaterbi/DrugPP/5jc5rieb &
# wandb agent luowaterbi/DrugPP/5jc5rieb &
# wandb agent luowaterbi/DrugPP/5jc5rieb
# wandb agent luowaterbi/DrugPP/vsy7j9cb &
# wandb agent luowaterbi/DrugPP/vsy7j9cb &
# wandb agent luowaterbi/DrugPP/vsy7j9cb &
# wandb agent luowaterbi/DrugPP/vsy7j9cb &
# wandb agent luowaterbi/DrugPP/vsy7j9cb

