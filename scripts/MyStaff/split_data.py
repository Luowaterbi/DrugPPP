import sys
import random

dataset = sys.argv[1]
train_rate = float(sys.argv[2])
val_rate = float(sys.argv[3])
path = "./data/" + dataset + "/"
input_path = path + dataset + "_reformated.txt"
all_data = []
with open(input_path, 'r') as reader:
    for line in reader:
        all_data.append(line)
title = all_data[0]
all_data = all_data[1:]
all_data_len = len(all_data)
train_num = int(all_data_len * train_rate) + 1
val_num = all_data_len - train_num if train_rate + val_rate == 1.0 else int(all_data_len * val_rate) + 1
for i in range(5):
    random.shuffle(all_data)
    train_path = path + "train_" + str(i) + ".csv"
    val_path = path + "val_" + str(i) + ".csv"
    test_path = path + "test_" + str(i) + ".csv"
    with open(train_path, 'w') as writer:
        writer.writelines([title] + all_data[:train_num])
    with open(val_path, 'w') as writer:
        writer.writelines([title] + all_data[train_num:train_num + val_num])
    if train_rate + val_rate < 1.0:
        with open(test_path, 'w') as writer:
            writer.writelines([title] + all_data[train_num + val_num:])
