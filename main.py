# coding: utf-8
"""
Author: Atma
"""

import warnings
import os
from os.path import join as p_join
import argparse
from rdkit import RDLogger, rdBase, Chem
import torch
import random
import json
import numpy as np

# local imports
from utils.data_loader import FloatTensor, load_raw_data, construct_loader
from utils.train import Trainer, make_optimizer_scheduler
from utils.test import Tester, rmse_metrics, acc_metrics
from model.model import SAIGNModel, make_model

DEFAULT_OUTPUT_DIR = './runs/'

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

# ====== task setting ======
parser.add_argument('--name', default='sagt', help="The name of the current running.")
parser.add_argument('--task', default='sol', choices=['ddi', 'sol'], help="the task of current running")
parser.add_argument('--output_dir', default='./runs/', help="dir of trained models")
parser.add_argument('--data_dir', default='./data/MNSOL/', help="dir of trained models")
parser.add_argument('--train_file', default='train_0.csv', help="train file path")
parser.add_argument('--valid_file', default='val_0.csv', help="valid file path")
parser.add_argument('--test_file', default='', help="test file path")
parser.add_argument('--y2id_file', default='', help="y2id dict path")

# ====== training setting ======
parser.add_argument('--seed', required=False, type=int, default=0, help="random seed")
parser.add_argument('--max_epochs', required=False, type=int, default=100, help="The max number of epochs for training")
parser.add_argument('--batch_size', required=False, type=int, default=32, help="The batch size for training")
parser.add_argument('--lr', required=False, type=float, default=0.005, help="")
parser.add_argument('--optimizer', required=False, default='adam', choices=['adam', 'adamw'], help="")
parser.add_argument('--weight_decay', required=False, type=float, default=0.1, help="")
parser.add_argument('--scheduler', required=False, default='plateau', choices=['plateau', 'lwp'], help="lwp: linear warm up, plateau: reduce on plateau.")
parser.add_argument('--warmup_proportion', required=False, type=float, default=0.05, help="rate for warm-up steps")
parser.add_argument('--patience', required=False, type=int, default=5, help="The batch size for training")
parser.add_argument('--load_param', required=False, default='none', choices=['pretrain', 'full', 'none'], help="not implement yet.")

# ====== model setting ======
parser.add_argument('--d_features', required=False, type=int, default=128, help="dim of atom features, ")
parser.add_argument('--d_model', required=False, type=int, default=128, help="dim of model's input and hidden layer")
parser.add_argument('--init_type', required=False, default='uniform', choices=['uniform', 'normal', 'small_normal_init', 'small_uniform_init'], help="different xavier")
parser.add_argument("--rn", default=False, action='store_true', help="and relation node in feature")
parser.add_argument("--rn_dst", required=False, type=float, default=1.0, help="default relation node distance to other node")
parser.add_argument("--cross_dst", required=False, type=float, default=1e6, help="default node distance between two graph")

# ====== encoder setting ======
parser.add_argument('--encoder', required=False, default='gt', choices=['gt', 'mlp'], help="gt: graph transformer.")
parser.add_argument('--enc_pair_type', required=False, default='sep', choices=['share', 'sep', 'joint'], help="share: share encoder for two input, " "sep: use different encoder for two input, " "joint: cat two input and send them to a single encoder")
parser.add_argument('--enc_n_layer', required=False, type=int, default=4, help="num of transformer layers")
parser.add_argument('--enc_n_head', required=False, type=int, default=4, help="num of attention heads")
parser.add_argument('--enc_dropout', required=False, type=float, default=0.1, help="dropout rate")
parser.add_argument('--lambda_attention', required=False, type=float, default=0.33, help="rate for MAT's attention")
parser.add_argument('--lambda_distance', required=False, type=float, default=0.33, help="rate for MAT's attention")
parser.add_argument('--trainable_lambda', required=False, default=False, action='store_true', help='')
parser.add_argument('--use_edge', required=False, default=False, action='store_true', help='use edge features')
parser.add_argument('--integrated_dst', required=False, default=False, action='store_true', help='if use edge, integrate distances mat into edge features')
parser.add_argument('--enc_scale_norm', required=False, default=False, action='store_true', help='true to use scale_norm else use LayerNorm')
parser.add_argument("--no_dummy", default=False, action='store_true', help="remove dummy node in GT.")

# ====== interactor setting ======
parser.add_argument('--interactor', required=False, default='simple', choices=['sa', 'rn', 'rnsa', 'none', 's1mple'], help="sa: self-attentive, rn: relation node, simple: GIGIN used")
parser.add_argument('--inter_n_layer', required=False, type=int, default=4, help="num of transformer layers")
parser.add_argument('--inter_n_head', required=False, type=int, default=4, help="num of attention heads")
parser.add_argument('--inter_dropout', required=False, type=float, default=0.1, help="dropout rate")
parser.add_argument('--inter_norm', required=False, default='layer_norm', choices=['layer_norm', 'none'], help="")
parser.add_argument('--type_emb', required=False, default='sep', choices=['sep', 'none'], help="different type emb for x1 and x2")
parser.add_argument('--att_block', required=False, default='none', choices=['none', 'self'], help="block transformer attention, self to only keep inter attention")
parser.add_argument('--inter_res', required=False, default='no_inter', choices=['cat', 'none', 'no_inter'], help="set residual connection between interaction's input and output")

# ====== MoE setting ======
parser.add_argument('--moe', required=False, default=1, type=int, help="whether use the MoE")
parser.add_argument('--mix', required=False, default=1, type=int, help="whether use the mix gate")
parser.add_argument('--moe_input', required=False, default='mol_avg', choices=['atom', 'mol_avg', 'mol_sum'], help="determine the experts based on atoms or molecule")
parser.add_argument('--noisy_gating', required=False, default=0, type=int, help="whether open the noisy gating")
parser.add_argument('--num_experts', required=False, type=int, default=32, help="the num of experts")
parser.add_argument('--num_used_experts', required=False, type=int, default=4, help="the num of used experts")
parser.add_argument('--moe_loss_coef', required=False, type=float, default=1e-1, help="the loss_load of MoE")
parser.add_argument('--moe_dropout', required=False, type=float, default=1e-1, help="the dropout prob of MoE")

# ====== decoder setting ======
parser.add_argument('--decoder', required=False, default='reg', choices=['reg', 'cls'], help="reg: regression decoder, cls: classification decoder.")
parser.add_argument('--readout', required=False, default='rn_sum', choices=['avg', 'set2set', 'rn', 'rn_avg', 'j_avg', 'rn_sum'], help="")

# ====== others ======
parser.add_argument("--debug", default=True, action='store_true', help="debug model, only load few data.")

opt = parser.parse_args()


random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

debug = opt.debug

# load_ft = True
load_ft = not opt.debug

train_path = os.path.join(opt.data_dir, opt.train_file)
valid_path = os.path.join(opt.data_dir, opt.valid_file)
test_path = os.path.join(opt.data_dir, opt.test_file) if opt.test_file else None
if opt.y2id_file:
    with open(os.path.join(opt.data_dir, opt.y2id_file)) as reader:
        dict_y2id = json.load(reader)
        opt.num_tags = len(dict_y2id)
else:
    dict_y2id = None


def main():
    train_x1, train_x2, train_joint, train_y = load_raw_data(opt.task, train_path, not opt.no_dummy, True, load_ft, opt.rn, opt.rn_dst, opt.cross_dst, dict_y2id, debug)
    valid_x1, valid_x2, valid_joint, valid_y = load_raw_data(opt.task, valid_path, not opt.no_dummy, True, load_ft, opt.rn, opt.rn_dst, opt.cross_dst, dict_y2id, debug)
    # print("train_num={},valid_num={}".format(len(train_y), len(valid_y)))
    if test_path:
        test_x1, test_x2, test_joint, test_y = load_raw_data(opt.task, test_path, not opt.no_dummy, True, load_ft, opt.rn, opt.rn_dst, opt.cross_dst, dict_y2id, debug)

    if debug:
        train_num, valid_num, opt.max_epochs, opt.batch_size, opt.name = 100, 100, 50, 32, '[debug]' + opt.name
        train_x1, train_x2, train_joint, train_y = [d[:train_num] for d in (train_x1, train_x2, train_joint, train_y)]
        valid_x1, valid_x2, valid_joint, valid_y = [d[:valid_num] for d in (valid_x1, valid_x2, valid_joint, valid_y)]
        if test_path:
            test_x1, test_x2, test_joint, test_y = [d[:valid_num] for d in (test_x1, test_x2, test_joint, test_y)]

    opt.d_features = train_x1[0][0].shape[1]  # It depends on the featurization of atom.

    train_loader = construct_loader(train_x1, train_x2, train_joint, train_y, opt.batch_size, joint_enc=(opt.enc_pair_type == 'joint'))
    valid_loader = construct_loader(valid_x1, valid_x2, valid_joint, valid_y, opt.batch_size, shuffle=False, joint_enc=(opt.enc_pair_type == 'joint'))
    if test_path:
        test_loader = construct_loader(test_x1, test_x2, test_joint, test_y, opt.batch_size, shuffle=False, joint_enc=(opt.enc_pair_type == 'joint'))
    else:
        test_loader = None

    print(opt)
    model = make_model(opt)
    model.to(device)

    optimizer, scheduler = make_optimizer_scheduler(opt, model, int(len(train_x1) / opt.batch_size * opt.max_epochs))
    loss_fn = torch.nn.MSELoss() if opt.decoder == 'reg' else torch.nn.CrossEntropyLoss()
    if opt.task == 'sol':
        better_score_f = lambda x, y: x < y
        metric_f = rmse_metrics
    else:
        better_score_f = lambda x, y: x > y
        metric_f = acc_metrics

    trainer = Trainer()
    tester = Tester(metric_func=metric_f)

    # train & test
    trainer.train(opt.max_epochs, model, optimizer, scheduler, opt.name, opt.output_dir, tester, loss_fn, train_loader, valid_loader, test_loader, True, opt.scheduler, better_score_f)


if __name__ == '__main__':
    main()
