from tqdm import tqdm
import torch
import numpy as np
from torch.nn import MSELoss
import math

flatten = lambda t: [item for sublist in t for item in sublist]


class Tester:
    def __init__(self, metric_func):
        self.metric_func = metric_func

    def test(self, model, data_loader):
        model.eval()
        all_pred, all_label = [], []
        for batch in data_loader:
            inputs, label = batch[:-1], batch[-1]
            pred = model(inputs)
            all_pred.append(pred)
            all_label.append(label)
        return self.metric_func(flatten(all_pred), flatten(all_label))


def rmse_metrics(all_pred, all_label):
    """
    all_pred: List[torch.tensor], list of each batch's model predication
    all_label: List[torch.tensor], , list of each batch's golden annotation

    return rmse, mse
    """
    loss_f = torch.nn.MSELoss()
    mse = loss_f(torch.tensor(all_pred), torch.tensor(all_label))
    loss_f2 = torch.nn.L1Loss()
    mae = loss_f2(torch.tensor(all_pred), torch.tensor(all_label))
    return math.sqrt(mse), mse, mae, torch.tensor(all_pred).tolist()


def acc_metrics(all_pred, all_label):
    """
    all_pred: List[torch.tensor], list of each batch's model predication
    all_label: List[torch.tensor], , list of each batch's golden annotation

    return acc, loss
    """
    correct = 0.0
    pred_label_lst = []
    # print(f'debug pred {all_pred.shape} {all_pred}, label {all_label.shape} {all_label}')
    for pred, label in zip(all_pred, all_label):
        pred_label = int(torch.argmax(pred))
        # print(f'debug pred {pred} {pred.shape}, label {label.shape} {label}')
        if pred_label == int(label):
            correct += 1
        pred_label_lst.append(pred_label)
    loss = 233
    # loss = torch.nn.functional.cross_entropy(torch.tensor(all_pred), torch.LongTensor(all_label))
    return correct / len(all_pred), loss, pred_label_lst
