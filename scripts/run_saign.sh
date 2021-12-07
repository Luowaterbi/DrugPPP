#!/usr/bin/env bash
echo usage: pass gpu id list as param, split with ,
echo eg: source run_bert_siamese.sh 3,4 stanford

gpu_list=$1

# Comment one of follow 2 to switch debugging status
#debug=--debug
debug=

#restore=--restore_cpt
restore=

# ======= dataset setting ======
dataset_lst=(MNSol)

# ====== train & test setting ======
#seed_lst=(0)
seed_lst=(0 1 2 3 4)
#seed_lst=(6150 6151 6152)

#lr_lst=(0.001 0.003)
#lr_lst=(0.003)
#lr_lst=(0.003)
lr_lst=(0.005)
#lr_lst=(0.003)
#lr_lst=(0.004)
#lr_lst=(0.005)
#lr_lst=(0.0005)

#train_batch_size_lst=(128)
train_batch_size_lst=(64)
#train_batch_size_lst=(32 64)
#train_batch_size_lst=(64 128)

#epoch=1000
epoch=200

#optimizer=adam
optimizer=adamw
#weight_decay_lst=(0.0001)
#weight_decay_lst=(0.01)
weight_decay_lst=(0.1)

scheduler=lwp
#scheduler=plateau

warmup_proportion_lst=(0.05)

patience=10

# ==== model setting =========
# ---- encoder setting -----
sep_emb=--sep_emb
#sep_emb=

res_type_lst=(all)
#res_type_lst=(interact)
#res_type_lst=(graph)
#res_type_lst=(all graph interact none)

tf_hidden_lst=(128)
#tf_hidden_lst=(64)
#tf_hidden_lst=(64 128 256)

#att_block_lst=(self)
att_block_lst=(none)

# ------ decoder setting -------

readout_lst=(set2set)
#readout_lst=(set2set avg)

num_tf_layer_lst=(2)
#num_tf_layer_lst=(3 2 1)

tf_norm_type_lst=(layer_norm)
#tf_norm_type_lst=(none)
#tf_norm_type_lst=(layer_norm none)

#tf_nhead_lst=(2)
tf_nhead_lst=(3)

dropout_lst=(0.1)
#dropout_lst=(0.3)
#dropout_lst=(0.5)
#dropout_lst=(0.3 0.5 0.1)

# ======= get into default path ==========


echo [START] set jobs on dataset [ ${dataset_lst[@]} ] on gpu [ ${gpu_list} ]
# === Loop for all case and run ===
for seed in ${seed_lst[@]}; do
  for dataset in ${dataset_lst[@]}; do
    for train_batch_size in ${train_batch_size_lst[@]}; do
      for dropout in ${dropout_lst[@]}; do
        for lr in ${lr_lst[@]}; do
          for tf_hidden in ${tf_hidden_lst[@]}; do
            for num_tf_layer in ${num_tf_layer_lst[@]}; do
              for res_type in ${res_type_lst[@]}; do
                for tf_norm_type in ${tf_norm_type_lst[@]}; do
                  for readout in ${readout_lst[@]}; do
                    for warmup_proportion in ${warmup_proportion_lst[@]}; do
                      for weight_decay in ${weight_decay_lst[@]}; do
                        for tf_nhead in ${tf_nhead_lst[@]}; do
                          for att_block in ${att_block_lst[@]}; do
                            # model names
                            model_name=saign_best.bs_${train_batch_size}.ep_${epoch}.optm_${optimizer}.wd_${weight_decay}.pa_${patience}.schd_${scheduler}.wp_${warmup_proportion}.lr_${lr}.inter_self_att.blc_att_${att_block}.n_tfl_${num_tf_layer}.tf_norm_${tf_norm_type}.tf_nhead_${tf_nhead}.tf_dropout_${dropout}.tf_mlp_${tf_hidden}.res_${res_type}.ro_${readout}.sep_${sep_emb}${debug}
                            file_mark=${dataset}.sd_${seed}

                            cd CIGIN_V2/

                            echo [CLI]
                            echo Model: ${model_name}
                            echo Task: ${file_mark}
                            echo [CLI]
                            export OMP_NUM_THREADS=3 # threads num for each task
                            CUDA_VISIBLE_DEVICES=${gpu_list} python main.py ${sep_emb} ${debug} \
                              --name ${model_name} \
                              --batch_size ${train_batch_size} \
                              --max_epochs ${epoch} \
                              --lr ${lr} \
                              --interaction self_att \
                              --num_tf_layer ${num_tf_layer} \
                              --tf_norm_type ${tf_norm_type} \
                              --readout ${readout} \
                              --tf_dropout ${dropout} \
                              --tf_mlp_dim ${tf_hidden} \
                              --res_connection ${res_type} \
                              --optimizer ${optimizer} \
                              --weight_decay ${weight_decay} \
                              --scheduler ${scheduler} \
                              --patience ${patience} \
                              --warmup_proportion ${warmup_proportion} \
                              --tf_nhead ${tf_nhead} \
                              --att_block ${att_block} \
                              --seed ${seed} >./log/${model_name}.DATA.${file_mark}.log
                            echo [CLI]
                            echo Model: ${model_name}
                            echo Task: ${file_mark}
                            echo [CLI]

                            cd ..

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
  done
done

echo [FINISH] set jobs on dataset [ ${dataset_lst[@]} ] on gpu [ ${gpu_list} ]
