#%%

import random
import pandas as pd
import pickle
import os
import sys
import warnings
from collections import OrderedDict
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit import RDLogger


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

from sklearn.metrics import mean_squared_error
from math import sqrt


sys.path.insert(0, './scripts/')
# from molecular_graph import ConstructMolecularGraph
from models import Cigin

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)

testing_path = 'CIGIN_V2/data/MNSol_reformated.txt'
model_path = '/users7/ythou/Projects/code/Medical/CIGIN/CIGIN_V2/runs/run-cigin/models/best_model.tar'


class Tester:
    def __init__(self):
        self.all_data = []

    def rmse(self, y_actual, y_predicted):
        return sqrt(mean_squared_error(y_actual, y_predicted))

    def testing(self, model):
        all_predict = []
        all_golden = []
        cnt = 0
        for data in self.all_data:
            solute_name, solvent_name, solute_smile, solvent_smile, energy = data
            delta_g, interaction_map = model(solute_smile, solvent_smile)
            all_golden.append(energy)
            all_predict.append(delta_g)
            cnt += 1
            if cnt % 10 == 0:
                print('{} / {} processed '.format(cnt, len(self.all_data)))
        print('RMSE is:', self.rmse(all_golden, all_predict))

    def load_data(self, path):
        with open(path, 'r') as reader:
            for line in reader:
                items = line.split(';')
                self.all_data.append(items)


def main():
    model = Cigin().to(device)
    model.load_state_dict(torch.load('weights/cigin.tar'))
    model.eval()

    tester = Tester()
    tester.load_data(testing_path)
    tester.testing(model)


if __name__ == '__main__':
    main()
