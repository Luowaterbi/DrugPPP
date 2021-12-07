# coding: utf-8
import os
from math import sqrt
import numpy as np

log_dir = "./log/"
# log_dir = "./CIGIN_V2/log"  # for old models
# log_dir = "/users5/ythou/Projects/code/Medical/CIGIN/CIGIN_V2/log"

mark_lst = ['rnign_inter_tune.']
# mark_lst = ['rnign_inter_more_epoch.']
# mark_lst = ['new_saign_.']
# mark_lst = ['rnign_inter.readout_rn.']
# mark_lst = ['rnign_inter.readout_rn_avg.']
# mark_lst = ['new_rnign_tf_only_correct_more_e_l']
# mark_lst = ['new_rnign_tf_only_try_read_out.']
# mark_lst = ['new_rnign_tf_only_more_e_l']
# mark_lst = ['new_rnign_tf_only.r']
# mark_lst = ['new_rnign_no_dummy.']
# mark_lst = ['new_saign_no_dummy_node.']
# mark_lst = ['new_rnign.readout_rn_avg', 'lambda_0.33_0.33',]
# mark_lst = ['new_rnign.readout_rn_avg', 'lambda_0.33_0.33', 'rn_dst_0', 'cr_dst_100']
# mark_lst = ['new_rnign.readout_rn_avg', 'lambda_0.33_0.33', 'rn_dst_1', 'cr_dst_100']
# mark_lst = ['new_rnign.readout_rn_avg', 'lambda_0.33_0.33', 'rn_dst_0', 'cr_dst_1e6']
# mark_lst = ['new_rnign.readout_rn_avg', 'lambda_0.33_0.33', 'rn_dst_1', 'cr_dst_1e6']

mark_type = 'and'
# mark_type = 'or'


ignore_lst = ['debug']
# ignore_lst = ['debug', 'tf_nhead_6', 'old_best']
# ignore_lst = ['debug', 'tf_nhead_6']


def mark_filter(item):
    for mark in ignore_lst:
        if mark in item:
            return False
    if mark_type == 'or':
        for mark in mark_lst:
            if mark in item:
                return True
    else:
        for mark in mark_lst:
            if mark not in item:
                return False
        return True
    return False


def CIGIN_extract_result_from_line(line):
    line = line.replace('MAE Val_loss ', '').replace('train_loss', '').replace('Val_loss', '')
    items = line.split()
    epoch = float(items[1])
    train_mse = float(items[2])
    val_mse = float(items[3])
    rmse = sqrt(val_mse)
    return epoch, train_mse, val_mse, rmse


def extract_result_from_line(line):
    line = line.replace('Epoch:', '').replace('Train Loss:', '').replace('Val Loss:', '').replace('Val Score:', '')
    items = line.split()
    epoch = float(items[0])
    train_mse = float(items[1])
    val_mse = float(items[2])
    rmse = float(items[3])
    return epoch, train_mse, val_mse, rmse


def collect_results():
    all_files = os.listdir(log_dir)
    all_best = 0
    all_result = []
    # print(all_files)
    for file in filter(mark_filter, all_files):
        with open(os.path.join(log_dir, file), 'r') as reader:
            best_val = 10000
            results = []
            for line in reader:
                # print('train_loss' in line, "Val_loss" in line, line)
                if ('train_loss' in line and "Val_loss" in line) or 'Train Loss:' in line:
                    epoch, train_mse, val_mse, rmse = extract_result_from_line(line)
                    # print(line, epoch, train_mse, val_mse, rmse)
                    # epoch, train_mse, val_mse, rmse = CIGIN_extract_result_from_line(line)
                    results.append([epoch, train_mse, val_mse, rmse])
                    # print(results[-1])
                    if best_val > rmse:
                        best_val = rmse
            if best_val < 10000 and epoch > 40:
            # if best_val < 10000:
                all_result.append([file, best_val, results])
    return all_result


def statistic(all_result):
    all_setting = {}
    for result in all_result:
        model = result[0]
        rmse = result[1]
        train_mse = result[2][-1][1]
        for setting in model.replace('0.', '0*_').split('.'):
            if setting not in all_setting:
                all_setting[setting] = [[rmse, train_mse, model]]
            else:
                all_setting[setting].append([rmse, train_mse, model])

    all_stats = sorted(all_setting.items(), key=lambda x: x[0])
    for setting, scores in all_stats:
        pure_score = list(zip(*scores))[0]
        pure_train_score = list(zip(*scores))[1]
        # print(pure_train_score)
        print('\n{}\nbest:{}, \nworst: {}, \navg: {} \nstd: {} res_num: {} avg_train_loss {}'.format(setting.replace('0*_', '0.'), min(scores), max(scores), sum(pure_score)/len(scores), np.std(pure_score), len(pure_score), sum(pure_train_score)/len(scores)))


def show_res(all_result):
    all_result = sorted(all_result, key=lambda x: x[0])
    for res in all_result:
        print("\nRMSE: {}, \nModel: {}, \nfinal state:{}".format(res[1], res[0], res[2][-1]))


def main():
    all_result = collect_results()
    show_res(all_result)
    statistic(all_result)


if __name__ == "__main__":
    main()
