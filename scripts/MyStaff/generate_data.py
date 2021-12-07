# coding: utf-8
from pubchempy import get_compounds
import random
import os
import time

dataset_name = 'MNSOL'
raw_data_name = 'MNSol_alldata.txt'
# dataset_name = 'freesolv'
# raw_data_name = 'freesolv.csv'

dataset_path = '/users10/xzluo/DrugDP/DrugPP/data/' + dataset_name + '/'
raw_path = dataset_path + raw_data_name  # 原始数据
output_path = dataset_path + dataset_name + '_reformated.txt'  # 原始数据转化后的总数据
train_path = dataset_path + 'train.csv'  # 训练集
val_path = dataset_path + 'val.csv'  # 验证集
res_path = dataset_path + 'res.txt'  # 需要手动添加的内容
half_dict_path = dataset_path + 'half_dict.csv'
full_dict_path = dataset_path + 'full_dict.csv'


def clear_string(name):
    if name[0] == '"':
        return eval(name)
    else:
        return name


class DataGenerator:
    def __init__(self):
        self.all_smiles = {}
        self.cant_find = []

    # def construct_smile_dict(self, current_output):  # 将化学表达式与名称建立字典，返回字典
    #     with open(current_output, 'r') as reader:
    #         first_column = True
    #         for line in reader:
    #             if first_column:
    #                 first_column = False
    #                 continue
    #             items = line.split(';')
    #             solute_name, solvent_name, solute_smile, solvent_smile, energy = items
    #             if solute_name not in self.all_smiles:
    #                 self.all_smiles[solute_name] = solute_smile
    #             if solvent_name not in self.all_smiles:
    #                 self.all_smiles[solvent_name] = solvent_smile
    #     return self.all_smiles

    def construct_smile_dict(self, path):
        with open(path, 'r') as reader:
            for line in reader:
                items = line.split(';')
                name, smile = items[0], items[1].replace('\n', '')
                if name not in self.all_smiles:
                    self.all_smiles[name] = smile
        return self.all_smiles

    #  读取数据
    def load(self, path, name):
        ret = []
        first_column = True
        if name == "MNSOL":
            with open(path, 'r') as reader:
                for line in reader:
                    if first_column:
                        first_column = False
                        continue
                    items = line.split('\t')
                    solute_name = clear_string(items[2])
                    charge = items[5]
                    solvent_name = clear_string(items[9])
                    energy = items[10]
                    tp = items[11]
                    # 带电的不能算进去，因为会有多种表示；相对自由能也不能要
                    if charge != '0' or tp == 'rel':
                        continue
                    ret.append([solute_name, solvent_name, energy])
        if name == "freesolv":
            with open(path, 'r') as reader:
                for line in reader:
                    if first_column:
                        first_column = False
                        continue
                    items = line.split(',')
                    # prevent data like '"xxx"'(have "" at start)
                    ret.append(
                        [clear_string(items[0]), items[1].replace('\n', '')])
        return ret

    #  从pubchempy中获取smile
    def get_smile(self, name):
        if name in self.all_smiles:
            return self.all_smiles[name]
        if name in self.cant_find:
            return None
        compounds = get_compounds(name, 'name')
        compound = compounds[0] if compounds else None
        if compound:
            print("Get formula of " + name)
            self.all_smiles[name] = compound.isomeric_smiles
            return compound.isomeric_smiles
        else:
            print("Can't Get formula of " + name)
            self.cant_find.append(name)
            return None

    # 从pubchempy中获取name
    def get_name(self, smile):
        compounds = get_compounds(smile, 'smiles')
        compound = compounds[0] if compounds else None
        if compound:
            return compound.iupac_name
        else:
            print("Can't get name of " + smile)
            self.cant_find.append(smile)
            return None

    def reformat_data(self, raw_data, output_file, name):
        with open(output_file, 'w') as writer:
            writer.write(';'.join([
                'Solute', 'Solvent', 'SoluteSMILES', 'SolventSMILES',
                'DeltaGsolv'
            ]) + '\n')
            if name == "MNSOL":
                for items in raw_data:
                    try:
                        solute_name, solvent_name, energy = items
                        solute_smile = self.get_smile(solute_name)
                        solvent_smile = self.get_smile(solvent_name)
                        if solvent_smile and solute_smile:
                            writer.write(';'.join([
                                solute_name, solvent_name, solute_smile,
                                solvent_smile, energy
                            ]) + '\n')
                    except Exception as e:
                        print("!!!!!!!!!!!Error: ", e, items)
            if name == "freesolv":
                cnt = 1
                solvent_name = "water"
                solvent_smile = "O"
                for items in raw_data:
                    try:
                        solute_smile, energy = items
                        solute_name = self.get_name(solute_smile)
                        if solute_name:
                            writer.write(';'.join([
                                solute_name, solvent_name, solute_smile,
                                solvent_smile, energy
                            ]) + '\n')
                            print("{} is written!".format(cnt))
                            cnt = cnt + 1
                        else:
                            print(solute_smile + "written failed!")
                    except Exception as e:
                        print("!!!!!!!!!!!Error: ", e, items)


def split_data(path):  # 划分训练集与验证集
    train_rate = 0.9
    all_data = []
    with open(output_path, 'r') as reader:
        for line in reader:
            all_data.append(line)
    train_num = int(len(all_data) * train_rate) + 1  # 向上取整
    title = all_data[0]  # 列名
    all_data = all_data[1:]
    random.shuffle(all_data)
    with open(train_path, 'w') as writer:
        writer.writelines([title] + all_data[:train_num])
    with open(val_path, 'w') as writer:
        writer.writelines([title] + all_data[train_num:])


def gen_data():
    generator = DataGenerator()
    if dataset_name == "MNSOL":
        generator.construct_smile_dict(full_dict_path)
    raw_data = generator.load(raw_path, dataset_name)
    print("raw_data have " + str(len(raw_data)))
    generator.reformat_data(raw_data, output_path, dataset_name)
    if len(generator.cant_find) != 0:
        print('Save Res')
        print("can't find " + str(len(generator.cant_find)))
        with open(res_path, 'a') as writer:
            writer.write('\n' +
                         time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
                         '\n')
            for name in generator.cant_find:
                writer.write(name + '\n')
    # print(len(generator.all_smiles))
    # with open(full_dict_path, 'w') as writer:
    #     for x, y in generator.all_smiles.items():
    #         print(x, y)
    #         writer.write(';'.j oin([x, y]) + '\n')


def main():
    if os.path.isfile(raw_path):
        print("Get Raw Data!")
        # gen_data()
        split_data(output_path)
        print("Work Done!")
    else:
        print("Can't get raw data!")
        print("Work Failed!")


if __name__ == "__main__":
    main()
