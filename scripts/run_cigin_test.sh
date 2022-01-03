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
dataset_lst=(freesolv)

# ====== train & test setting ======
seed_lst=(0 1 2 3 4)
data_lst=(0 1 2)
# seed_lst=(6150 6151 6152)

#lr_lst=(0.001 0.003 0.0005)
lr_lst=(0.001)
#lr_lst=(0.005)
#lr_lst=(0.0005)

#train_batch_size_lst=(128)
train_batch_size_lst=(32)
#train_batch_size_lst=(64 128)

epoch=100

optimizer=adam

# ==== model setting =========
# ---- encoder setting -----
sep_emb=--sep_emb
#sep_emb=

readout_lst=(set2set)
#readout_lst=(set2set avg)
res_type_lst=(all)
#res_type_lst=(interact)
#res_type_lst=(graph)
#res_type_lst=(all graph interact none)

#tf_hidden_lst=(64 128)
tf_hidden_lst=(64)
#tf_hidden_lst=(64 128 256)

# ------ decoder setting -------
#decoder_lst=(rule)
num_tf_layer_lst=(1)
#num_tf_layer_lst=(2 1 3)

#tf_norm_type_lst=(layer_norm)
tf_norm_type_lst=(none)
#tf_norm_type_lst=(layer_norm none)

#tf_nhead_lst=(2)
tf_nhead_lst=(3)

dropout_lst=(0.1)
#dropout_lst=(0.3)
#dropout_lst=(0.5)
#dropout_lst=(0.3 0.5 0.1)

# ======= default path (for quick distribution) ==========
# data path
#base_data_dir=/users4/yklai/code/Dialogue/release/MetaDialog/data/smp/

echo [START] set jobs on dataset [ ${dataset_lst[@]} ] on gpu [ ${gpu_list} ]
# === Loop for all case and run ===
for dataset in ${dataset_lst[@]}; do
  for train_batch_size in ${train_batch_size_lst[@]}; do
    for dropout in ${dropout_lst[@]}; do
      for lr in ${lr_lst[@]}; do
        for tf_hidden in ${tf_hidden_lst[@]}; do
          for num_tf_layer in ${num_tf_layer_lst[@]}; do
            for res_type in ${res_type_lst[@]}; do
              for tf_norm_type in ${tf_norm_type_lst[@]}; do
                for readout in ${readout_lst[@]}; do
                  for tf_nhead in ${tf_nhead_lst[@]}; do
                    # model names
                    model_name=cigin.baseline.no_reg.bs_${train_batch_size}.ep_${epoch}.lr_${lr}.inter_dot.res_${res_type}.ro_${readout}${debug}
                    # model_name=saign.bs_${train_batch_size}.ep_${epoch}.optm_${}.lr_${lr}.inter_self_att.n_tfl_${num_tf_layer}.tf_norm_${tf_norm_type}.tf_nhead_${tf_nhead}.tf_dropout_${dropout}.tf_mlp_${tf_hidden}.res_${res_type}.ro_${readout}.sep_${sep_emb}${debug}.old_split
                    # model_name=cigin_output_mid
                    cd CIGIN_V2
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
                        CUDA_VISIBLE_DEVICES=${gpu_list} python main.py ${sep_emb} ${debug} \
                          --name ${model_name}_${file_mark} \
                          --data_dir ../data/${dataset} \
                          --output_dir ./runs/${dataset}/${model_name}/${file_mark}.model/ \
                          --train_file train_${data}.csv \
                          --valid_file val_${data}.csv \
                          --test_file test_${data}.csv \
                          --batch_size ${train_batch_size} \
                          --max_epochs ${epoch} \
                          --lr ${lr} \
                          --interaction dot \
                          --warmup_proportion 0.05 \
                          --num_tf_layer ${num_tf_layer} \
                          --tf_norm_type ${tf_norm_type} \
                          --readout ${readout} \
                          --tf_dropout ${dropout} \
                          --tf_mlp_dim ${tf_hidden} \
                          --res_connection ${res_type} \
                          --tf_nhead ${tf_nhead} \
                          --seed ${seed} >./log/${dataset}/${model_name}/${file_mark}.log
                        echo [CLI]
                        echo Model: ${model_name}
                        echo Task: ${file_mark}
                        echo [CLI]  
                      done
                    done
                    cd ..
                    python scripts/MyStaff/cal_avg.py ./CIGIN_V2/runs/${dataset}/${model_name}/best.txt
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
