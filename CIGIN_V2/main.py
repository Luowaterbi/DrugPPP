# python imports
import pandas as pd
import warnings
import os
import argparse

# rdkit imports
from rdkit import RDLogger
from rdkit import rdBase
from rdkit import Chem

# torch imports
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

#dgl imports
import dgl

# local imports
from model import CIGINModel, SAIGNModel
from train import train
from molecular_graph import get_graph_from_smile
from utils import *

# my imports
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np

default_output_dir = './runs/'
# default_output_dir = '/users4/ythou/Projects/Medical/CIGIN/runs/'

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='cigin', help="The name of the current project: default: CIGIN")
parser.add_argument('--output_dir', default='./runs/', help="dir of trained models")
parser.add_argument('--data_dir', default='../data/MNSOL/', help="dir of data")
parser.add_argument('--train_file', default='train_0.csv', help="train file path")
parser.add_argument('--valid_file', default='val_0.csv', help="valid file path")
parser.add_argument('--test_file', required=False, default='', help="test file path")
parser.add_argument('--interaction', help="type of interaction function to use: dot | scaled-dot | general | "
                    "tanh-general | self_att | none ", default='dot')
parser.add_argument('--seed', required=False, type=int, default=0, help="")

parser.add_argument('--max_epochs', required=False, type=int, default=100, help="The max number of epochs for training")
parser.add_argument('--batch_size', required=False, type=int, default=32, help="The batch size for training")
parser.add_argument('--lr', required=False, type=float, default=0.001, help="")
parser.add_argument('--optimizer', required=False, default='adam', choices=['adam', 'adamw'], help="")
parser.add_argument('--weight_decay', required=False, type=float, default=0.001, help="")
parser.add_argument('--scheduler', required=False, default='plateau', choices=['plateau', 'lwp'], help="")
parser.add_argument('--warmup_proportion', required=False, type=float, default=0.1, help="rate for warm-up steps")
parser.add_argument('--patience', required=False, type=int, default=5, help="The batch size for training")

parser.add_argument('--tf_dropout', required=False, type=float, default=0.1, help="")
parser.add_argument('--tf_norm_type', required=False, default='layer_norm', help="")
parser.add_argument('--tf_dim', required=False, type=int, default=128, help="")
parser.add_argument('--tf_mlp_dim', required=False, type=int, default=128, help="")
parser.add_argument('--num_tf_layer', required=False, type=int, default=3, help="")
parser.add_argument('--tf_nhead', required=False, type=int, default=3, help="")
parser.add_argument('--att_block', required=False, default='none', choices=['none', 'self'], help="block transformer attention, self to only keep inter attention")

parser.add_argument(
    '--res_connection',
    required=False,
    default='all',
    choices=['all', 'graph', 'interact', 'none'],
    help="",
)
parser.add_argument('--readout', required=False, default='set2set', help="")
parser.add_argument('--pred_layer', required=False, default='linear', choices=['linear', 'bilinear'], help="")
parser.add_argument('--sep_emb', required=False, default=False, action='store_true', help='')

parser.add_argument('--load_param', required=False, default='cigin', choices=['cigin', 'ours'], help="")

parser.add_argument("--debug", default=False, action='store_true', help="debug model, only load few data.")

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def collate(samples):
    solute_graphs, solvent_graphs, labels = map(list, zip(*samples))
    solute_graphs = dgl.batch(solute_graphs)
    solvent_graphs = dgl.batch(solvent_graphs)
    # print("debug!!!", solute_graphs.batch_num_nodes)
    solute_len_matrix = get_len_matrix(solute_graphs.batch_num_nodes())
    solvent_len_matrix = get_len_matrix(solvent_graphs.batch_num_nodes())
    return solute_graphs, solvent_graphs, solute_len_matrix, solvent_len_matrix, labels


# class Dataclass(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#
#         solute = self.dataset.loc[idx]['SoluteSMILES']
#         mol = Chem.MolFromSmiles(solute)
#         mol = Chem.AddHs(mol)
#         solute = Chem.MolToSmiles(mol)
#         solute_graph = get_graph_from_smile(solute)
#
#         solvent = self.dataset.loc[idx]['SolventSMILES']
#         mol = Chem.MolFromSmiles(solvent)
#         mol = Chem.AddHs(mol)
#         solvent = Chem.MolToSmiles(mol)
#
#         solvent_graph = get_graph_from_smile(solvent)
#         delta_g = self.dataset.loc[idx]['DeltaGsolv']
#         return [solute_graph, solvent_graph, [delta_g]]


class Dataclass(Dataset):

    def __init__(self, raw_dataset):
        self.dataset = []
        for idx in range(len(raw_dataset)):
            solute = raw_dataset.loc[idx]['SoluteSMILES']
            # solute_name = raw_dataset.loc[idx]['Solute']
            mol = Chem.MolFromSmiles(solute)
            mol = Chem.AddHs(mol)
            solute = Chem.MolToSmiles(mol)
            solute_graph = get_graph_from_smile(solute)

            solvent = raw_dataset.loc[idx]['SolventSMILES']
            # solvent_name = raw_dataset.loc[idx]['Solvent']
            mol = Chem.MolFromSmiles(solvent)
            mol = Chem.AddHs(mol)
            solvent = Chem.MolToSmiles(mol)
            solvent_graph = get_graph_from_smile(solvent)

            delta_g = raw_dataset.loc[idx]['DeltaGsolv']
            self.dataset.append([solute_graph, solvent_graph, [delta_g]])
            # if idx % 100 == 0:
            #     print('processing feature: {}/{} processed'.format(idx, len(raw_dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def main():
    # train_df = pd.read_csv('data/MNSol_reformated.txt', sep=";")
    # valid_df = pd.read_csv('data/MNSol_reformated.txt', sep=";")
    train_df = pd.read_csv(os.path.join(args.data_dir, args.train_file), sep=";")
    valid_df = pd.read_csv(os.path.join(args.data_dir, args.valid_file), sep=";")
    test_df = pd.read_csv(os.path.join(args.data_dir, args.test_file), sep=";") if args.test_file else None

    if args.debug:
        train_df = train_df[:20]
        valid_df = valid_df[:10]
        args.max_epochs = 2
        args.name = '[debug]' + args.name
        args.batch_size = 2

    train_dataset = Dataclass(train_df)
    valid_dataset = Dataclass(valid_df)
    test_dataset = Dataclass(test_df) if args.test_file else None

    # ========= Context ===========
    project_name = args.name
    interaction = args.interaction
    max_epochs = int(args.max_epochs)
    batch_size = int(args.batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # output_dir = '/users4/ythou/Projects/Medical/CIGIN/runs/-'
    output_dir = args.output_dir

    # ========== train & test ===========
    train_loader = DataLoader(train_dataset, collate_fn=collate, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=16)
    test_loader = DataLoader(test_dataset, collate_fn=collate, batch_size=16) if test_dataset else None

    if interaction in ['dot', 'scaled-dot', 'general', 'tanh-general']:
        model = CIGINModel(interaction=interaction)
    elif interaction in ['self_att', 'none']:
        model = SAIGNModel(
            node_input_dim=42,
            edge_input_dim=10,
            node_hidden_dim=42,
            edge_hidden_dim=42,
            num_step_message_passing=6,
            interaction=interaction,
            num_step_set2_set=2,
            num_layer_set2set=1,
            num_tf_layer=args.num_tf_layer,
            tf_norm_type=args.tf_norm_type,
            readout_type=args.readout,
            tf_dim=args.tf_dim,
            tf_nhead=args.tf_nhead,
            tf_mlp_dim=args.tf_mlp_dim,
            tf_dropout=args.tf_dropout,
            pred_layer=args.pred_layer,
            sep_emb=args.sep_emb,
            res_connection=args.res_connection,
            att_block=args.att_block,
        )
    else:
        raise ValueError("Wrong interaction!")
    model.to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    elif args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=float(args.lr), weight_decay=args.weight_decay, correct_bias=True)
    else:
        raise ValueError("Wrong choice for optimizer")

    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, mode='min', verbose=True)
    elif args.scheduler == "lwp":
        num_train_steps = int(len(train_dataset) / batch_size * max_epochs)
        num_warmup_steps = int(args.warmup_proportion * num_train_steps)
        # print("Debug!! train steps {}, warmup steps {}".format(num_train_steps, num_warmup_steps))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
    else:
        raise ValueError("Wrong choice for scheduler")
    print(args)
    train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, test_loader, project_name, output_dir)


if __name__ == '__main__':
    main()
