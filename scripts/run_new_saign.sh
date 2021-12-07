#!/usr/bin/env bash
echo usage: pass gpu id list as param, split with ,
echo eg: source run_bert_siamese.sh 3,4 stanford

gpu_list=$1

# Comment one of follow 2 to switch debugging status
#debug=--debug
debug=

# ======= dataset setting ======
dataset_lst=(MNSol)

# ====== train & test setting ======
#seed_lst=(0)
seed_lst=(0 1 2 3 4)

lr_lst=(0.001)

train_batch_size_lst=(64)

epoch=300

#optimizer=adam
optimizer=adamw

weight_decay_lst=(0.1)

scheduler=lwp
#scheduler=plateau

warmup_proportion_lst=(0.05)

patience=10

# ==== model setting =========
d_model_lst=(128)

init_lst=(uniform)
#init_lst=(uniform normal small_normal_init small_uniform_init)

# ---- encoder setting -----
#encoder=mlp
encoder=gt

enc_layer_lst=(4)

enc_head_lst=(4)

enc_dropout_lst=(0.1)

#enc_pair_type_lst=(share sep)
enc_pair_type_lst=(sep)

lambda_att=0.33
lambda_dst=0.33


# --- interact setting ---
interactor=sa
#interactor=none

type_emb_lst=(sep)
#type_emb_lst=(none)

inter_res_lst=(cat)

inter_layer_lst=(4)

inter_head_lst=(4)

inter_dropout_lst=(0.1)

inter_norm_type_lst=(layer_norm)
#inter_norm_type_lst=(none)

#att_block_lst=(self)
att_block_lst=(none)

# ------ decoder setting -------

readout_lst=(avg)
#readout_lst=(set2set)

# ======= get into default path ==========

echo [START] set jobs on dataset [ ${dataset_lst[@]} ] on gpu [ ${gpu_list} ]
# === Loop for all case and run ===
# shellcheck disable=SC2068
for seed in ${seed_lst[@]}; do
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
                                        # model names
                                        model_name=new_saign_.bs_${train_batch_size}.ep_${epoch}.lr_${lr}_.d_model_${d_model}.inter_${interactor}_l_${inter_layer}_h_${inter_head}.enc_${encoder}_l_${enc_layer}_h_${enc_head}${debug}
                                        file_mark=${dataset}.sd_${seed}

                                        echo [CLI]
                                        echo Model: ${model_name}
                                        echo Task: ${file_mark}
                                        echo [CLI]
                                        export OMP_NUM_THREADS=3 # threads num for each task
                                        CUDA_VISIBLE_DEVICES=${gpu_list} python main.py ${debug} \
                                          --name ${model_name} \
                                          --batch_size ${train_batch_size} \
                                          --max_epochs ${epoch} \
                                          --lr ${lr} \
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
                                          --readout ${readout} \
                                          --optimizer ${optimizer} \
                                          --weight_decay ${weight_decay} \
                                          --scheduler ${scheduler} \
                                          --patience ${patience} \
                                          --warmup_proportion ${warmup_proportion} \
                                          --seed ${seed} >./log/${model_name}.DATA.${file_mark}.log
                                        echo [CLI]
                                        echo Model: ${model_name}
                                        echo Task: ${file_mark}
                                        echo [CLI]

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
