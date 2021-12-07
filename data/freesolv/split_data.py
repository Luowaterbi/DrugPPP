import math
import random

train_rate = 0.8
valid_rate = 0.1
test_rate = 0.1
data_path = "freesolv_reformated.txt"
train_path = "train.csv"
valid_path = "val.csv"
test_path = "test.csv"


def main():
    with open(data_path, 'r') as reader:
        data = [line for line in reader]
        title = data[0]
        data = data[1:]
        data_len = len(data)
        train_num = math.ceil(data_len * train_rate)
        test_num = math.ceil(data_len * test_rate)
        random.shuffle(data)
        with open(train_path, 'w') as writer:
            writer.writelines([title] + data[:train_num])
        with open(valid_path, 'w') as writer:
            writer.writelines([title] + data[train_num:-test_num])
        with open(test_path, 'w') as writer:
            writer.writelines([title] + data[-test_num:])


if __name__ == "__main__":
    main()