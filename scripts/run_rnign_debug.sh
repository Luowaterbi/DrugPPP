#!/usr/bin/env bash
echo usage: pass gpu id list as param, split with ,
echo eg: source run_bert_siamese.sh 3,4 stanford

gpu_list=$1

# Comment one of follow 2 to switch debugging status
# debug=--debug
debug=

# ======= dataset setting ======
dataset_lst=MNSOL

# ====== train & test setting ======
if [ $dataset_lst = MNSOL ]; then
  # seed_lst=(0 2 4)
  # data_lst=(0 1 2)
  seed_lst=(0)
  data_lst=(0)
elif [ $dataset_lst = TUM+Free ]; then
  seed_lst=(42 100)
  data_lst=(0 2 3 4)
elif [ $dataset_lst = esol ]; then
  seed_lst=(42 100 1000)
  data_lst=(0 1 2 3 4)
else
  echo NO THIS DATASET
fi

lr_lst=(0.0005)
# lr_lst=(0.001)
# lr_lst=(0.001 0.003 0.0005)

train_batch_size_lst=(32)

epoch=300

optimizer=adam
# optimizer=adamw

weight_decay_lst=(0.1)

scheduler=lwp
#scheduler=plateau

warmup_proportion_lst=(0.05)

patience=10

# ==== model setting =========
d_model_lst=(128)

init_lst=(uniform)
#init_lst=(uniform normal small_normal_init small_uniform_init)

rn_dst_lst=(1)
#rn_dst_lst=(0 1)

cross_dst_lst=(1e6)
#cross_dst_lst=(1e6 100)

# ---- encoder setting -----
#encoder=mlp
encoder=gt

enc_layer_lst=(4)

enc_head_lst=(4)

enc_dropout_lst=(0.1)

#enc_pair_type_lst=(share sep joint)
enc_pair_type_lst=(sep)
#enc_pair_type_lst=(joint)

#lambda_att=0.2
lambda_att=0.33
#lambda_dst=0.3
lambda_dst=0.33


# --- interact setting ---
interactor=$2
# interactor=simple
#interactor=rn
#interactor=sa
#interactor=none

type_emb_lst=(sep)
#type_emb_lst=(none)

inter_res_lst=(no_inter)
#inter_res_lst=(cat)
# 有MoE的时候，Mix Gate输入交互后的feature
# inter_res_lst=(none)

inter_layer_lst=(4)

inter_head_lst=(4)

inter_dropout_lst=(0.1)

inter_norm_type_lst=(layer_norm)
#inter_norm_type_lst=(none)

#att_block_lst=(self)
att_block_lst=(none)

# ------ MoE setting --------
moe_lst=(1)
mix=1
# moe_input_lst=(atom)  
moe_input_lst=$3
noisy_gating=1
num_experts_lst=(32)
num_used_experts_lst=(4)
moe_loss_coef_lst=(1e-2)
moe_dropout=1e-1


# ------ decoder setting -------

# readout_lst=(rn_sum rn rn_avg)
readout_lst=(rn_sum)
# readout_lst=(set2set)
#readout_lst=(avg)
#readout_lst=(set2set)

# ======= get into default path ==========

echo [START] set jobs on dataset [ ${dataset_lst[@]} ] on gpu [ ${gpu_list} ]
# === Loop for all case and run ===
# shellcheck disable=SC2068
for dataset in ${dataset_lst[@]}; do
  for train_batch_size in ${train_batch_size_lst[@]}; do
    for enc_dropout in ${enc_dropout_lst[@]}; do
      for inter_dropout in ${inter_dropout_lst[@]}; do
        for lr in ${lr_lst[@]}; do
          for d_model in ${d_model_lst[@]}; do
            for init in ${init_lst[@]}; do
              for enc_layer in ${enc_layer_lst[@]}; do
                for enc_pair_type in ${enc_pair_type_lst[@]}; do
                  for inter_layer in ${inter_layer_lst[@]}; do
                    for inter_res in ${inter_res_lst[@]}; do
                      for inter_norm in ${inter_norm_type_lst[@]}; do
                        for type_emb in ${type_emb_lst[@]}; do
                          for readout in ${readout_lst[@]}; do
                            for warmup_proportion in ${warmup_proportion_lst[@]}; do
                              for weight_decay in ${weight_decay_lst[@]}; do
                                for enc_head in ${enc_head_lst[@]}; do
                                  for inter_head in ${inter_head_lst[@]}; do
                                    for att_block in ${att_block_lst[@]}; do
                                      for rn_dst in ${rn_dst_lst[@]}; do
                                        for cross_dst in ${cross_dst_lst[@]}; do
                                          for num_experts in ${num_experts_lst[@]}; do
                                            for num_used_experts in ${num_used_experts_lst[@]}; do
                                              for moe_loss_coef in ${moe_loss_coef_lst[@]}; do
                                                for moe in ${moe_lst[@]}; do
                                                  for moe_input in ${moe_input_lst[@]}; do

                                                    if [ $moe -eq 1 ]; then
                                                      compare=
                                                    else
                                                      compare=--compare
                                                    fi

                                                    model_name=rnign.0107.test_wandb.interactor_${interactor}.moe_input_${moe_input}${debug}${compare}
                                                    # model_name=rnign.overfit.right.readout_${readout}.bs_${train_batch_size}.ep_${epoch}.lr_${lr}.warmup_${warmup_proportion}${debug}${compare}
                                                    # model_name=rnign.0106.test_new_moe_with_moe_loss
                                                    runsdir=./runs/${dataset}/${model_name}
                                                    logdir=./log/${dataset}/${model_name}
                                                    
                                                    if [ ! -d ${runsdir} ]; then
                                                      mkdir ${runsdir}
                                                    fi 
                                                    if [ ! -d ${logdir} ]; then
                                                      mkdir ${logdir}
                                                    fi 

                                                    for data in ${data_lst[@]}; do
                                                      for seed in ${seed_lst[@]}; do

                                                        file_mark=${dataset}_${data}.sd_${seed}
                                                        modeldir=./runs/${dataset}/${model_name}/${file_mark}.model
                                                        if [ ! -d ${modeldir} ];then
                                                          mkdir ${modeldir}
                                                        fi
                                                        echo [CLI]
                                                        echo Model: ${model_name}
                                                        echo Task: ${file_mark}
                                                        echo [CLI]
                                                        export OMP_NUM_THREADS=3 # threads num for each task
                                                        CUDA_VISIBLE_DEVICES=${gpu_list} python main.py ${debug} \
                                                          --name ${model_name}_${file_mark} \
                                                          --data_dir ./data/${dataset} \
                                                          --output_dir ./runs/${dataset}/${model_name}/${file_mark}.model/ \
                                                          --train_file train_${data}.csv \
                                                          --valid_file val_${data}.csv \
                                                          --batch_size ${train_batch_size} \
                                                          --max_epochs ${epoch} \
                                                          --lr ${lr} \
                                                          --rn_dst ${rn_dst} \
                                                          --cross_dst ${cross_dst} \
                                                          --d_model ${d_model} \
                                                          --init_type ${init} \
                                                          --encoder ${encoder} \
                                                          --enc_n_layer ${enc_layer} \
                                                          --enc_n_head ${enc_head} \
                                                          --enc_dropout ${enc_dropout} \
                                                          --enc_pair_type ${enc_pair_type} \
                                                          --lambda_attention ${lambda_att} \
                                                          --lambda_distance ${lambda_dst} \
                                                          --interactor ${interactor} \
                                                          --inter_n_layer ${inter_layer} \
                                                          --inter_n_head ${inter_head} \
                                                          --inter_dropout ${inter_dropout} \
                                                          --inter_norm ${inter_norm} \
                                                          --inter_res ${inter_res} \
                                                          --type_emb ${type_emb} \
                                                          --att_block ${att_block} \
                                                          --moe ${moe} \
                                                          --mix ${mix} \
                                                          --moe_input ${moe_input} \
                                                          --noisy_gating ${noisy_gating} \
                                                          --num_experts ${num_experts} \
                                                          --num_used_experts ${num_used_experts} \
                                                          --moe_loss_coef ${moe_loss_coef} \
                                                          --moe_dropout ${moe_dropout} \
                                                          --readout ${readout} \
                                                          --optimizer ${optimizer} \
                                                          --weight_decay ${weight_decay} \
                                                          --scheduler ${scheduler} \
                                                          --patience ${patience} \
                                                          --warmup_proportion ${warmup_proportion} \
                                                          --seed ${seed} >./log/${dataset}/${model_name}/${file_mark}.log
                                                        echo [CLI]
                                                        echo Model: ${model_name}
                                                        echo Task: ${file_mark}
                                                        echo [CLI]
                                                      done
                                                    done
                                                    python scripts/MyStaff/cal_avg.py ./runs/${dataset}/${model_name}/best.txt
                                                  done
                                                done
                                              done
                                            done
                                          done
                                        done
                                      done
                                    done
                                  done
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo [FINISH] set jobs on dataset [ ${dataset_lst[@]} ] on gpu [ ${gpu_list} ]
