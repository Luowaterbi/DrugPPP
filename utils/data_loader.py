"""
Author: Atma
Code based on:
Maziarka Ł, Danel T, Mucha S, et al. "Molecule Attention Transformer" -> https://github.com/ardigen/MAT/
Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks" -> https://github.com/Luckick/EAGCN
Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction" -> https://github.com/connorcoley/conv_qsar_fast
"""

import logging
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import json
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset

MAX_ATTEMPTS = 500
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def get_feature_stamp(add_dummy_node, one_hot_formal_charge, add_relation_node, rn_dst, cross_dst):
    dn_stamp = "_dn" if add_dummy_node else ""
    ohfc_stamp = "_ohfc" if one_hot_formal_charge else ""
    rn_stamp = "_rn_{}_{}".format(rn_dst, cross_dst) if add_relation_node else ""
    attempts_stamp = f"_ma_{MAX_ATTEMPTS}"
    return ''.join([dn_stamp, ohfc_stamp, rn_stamp, attempts_stamp])


def load_raw_data(task, dataset_path, add_dummy_node=True, one_hot_formal_charge=True, use_data_saving=True,
                  add_relation_node=True, rn_dst=1, cross_dst=1e6, dict_y2id=None, debug=False):
    """Load and featurize data stored in a CSV file.

    Args:
        dataset_path (str): A path to the CSV file containing the data. It should have two columns:
                            the first one contains SMILES strings of the compounds,
                            the second one contains labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.
        use_data_saving (bool): If True, saved features will be loaded from the dataset directory; if no feature file
                                is present, the features will be saved after calculations. Defaults to True.

    Returns:
        A tuple (x1, x2, y):
            x1, x2 are lists of graph descriptors (node features, adjacency matrices, distance matrices),
            y is a list of the corresponding labels.
    """
    feat_stamp = get_feature_stamp(add_dummy_node, one_hot_formal_charge, add_relation_node, rn_dst, cross_dst)
    feature_path = dataset_path + f'{feat_stamp}.p'
    if use_data_saving and os.path.exists(feature_path):
        logging.info(f"Loading features stored at '{feature_path}'")
        x1_all, x2_all, joint_all, y_all = pickle.load(open(feature_path, "rb"))
        return x1_all, x2_all, joint_all, y_all

    if task == 'sol':
        data_x1, data_x2, data_y = load_sol_raw_data(dataset_path)
    elif task == 'ddi':
        data_x1, data_x2, data_y = load_ddi_raw_data(dataset_path, dict_y2id)
    else:
        raise NotImplementedError

    x1_all = load_data_from_smiles(data_x1, add_dummy_node, one_hot_formal_charge, debug)
    x2_all = load_data_from_smiles(data_x2, add_dummy_node, one_hot_formal_charge, debug)
    y_all = [label for label in data_y]

    x1_all, x2_all, y_all = filter_out_bad_mol(x1_all, x2_all, y_all)

    joint_all = build_joint_features(
        x1_all, x2_all, add_dummy_node, one_hot_formal_charge, add_relation_node, rn_dst, cross_dst)

    if use_data_saving and not os.path.exists(feature_path):
        logging.info(f"Saving features at '{feature_path}'")
        pickle.dump((x1_all, x2_all, joint_all, y_all), open(feature_path, "wb"))

    return x1_all, x2_all, joint_all, y_all


def filter_out_bad_mol(x1_all, x2_all, y_all):
    clean_x1_all, clean_x2_all, clean_y_all = [], [], []
    for x1, x2, y in zip(x1_all, x2_all, y_all):
        if x1 and x2 and y:
            clean_x1_all.append(x1)
            clean_x2_all.append(x2)
            clean_y_all.append(y)
    return clean_x1_all, clean_x2_all, clean_y_all


def load_sol_raw_data(dataset_path):
    """ load MNSol data  """
    data_df = pd.read_csv(dataset_path, sep=";", engine="python")

    # columns are x1 name; x2 name, x1 smile, x2 smile, label
    data_x1 = data_df.iloc[:, 2].values
    data_x2 = data_df.iloc[:, 3].values
    data_y = data_df.iloc[:, 4].values

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)
    return data_x1, data_x2, data_y


def load_ddi_raw_data(dataset_path, dict_y2id):
    """ load drug-drug-interaction data """
    with open(dataset_path, 'r') as reader:
        json_data = json.load(reader)

    # json data: List[[d_id1, n1, s1, d_id2, n2, s2], y_itp]
    data_x1 = [i[0][2] for i in json_data]
    data_x2 = [i[0][5] for i in json_data]
    # print('debug', [i[1] for i in json_data])
    # print('debug dict~~~~~~~~~ \n', dict_y2id)
    data_y = [dict_y2id[i[1]] for i in json_data]
    return data_x1, data_x2, data_y


def load_data_from_smiles(x_smiles, add_dummy_node=True, one_hot_formal_charge=False, debug=False):
    """Load and featurize data from lists of SMILES strings and labels.

    Args:
        x_smiles (list[str]): A list of SMILES strings.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.

    Returns:
        a list of graph descriptors (node features, adjacency matrices, distance matrices).
    """
    x_all = []
    tq_loader = tqdm(x_smiles)
    if os.path.exists('./smile_cache.pk'):
        with open('./smile_cache.pk', 'rb') as reader:
            cache = pickle.load(reader)
    else:
        cache = {}
    for ind, smiles in enumerate(tq_loader):
        if smiles in cache:
            x_all.append(cache[smiles])
            continue
        try:
            mol = MolFromSmiles(smiles)
            if mol is None or mol == None:
                # print('debug', smiles)
                cache[smiles] = None
                x_all.append(None)
                continue
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=MAX_ATTEMPTS)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:
                AllChem.Compute2DCoords(mol)

            afm, adj, dist = featurize_mol(mol, add_dummy_node, one_hot_formal_charge)
            cache[smiles] = [afm, adj, dist, smiles]
            x_all.append([afm, adj, dist, smiles])
        except ValueError as e:
            logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

        tq_loader.set_description('{}/{} feature processed.'.format(ind, len(x_smiles)))
        if debug and ind > 20:
            break
        if ind % 1000 == 0:
            with open('./smile_cache.pk', 'wb') as writer:
                pickle.dump(cache, writer)
    with open('./smile_cache.pk', 'wb') as writer:
        pickle.dump(cache, writer)
    return x_all


def featurize_mol(mol, add_dummy_node, one_hot_formal_charge):
    """Featurize molecule.

    Args:
        mol (rdchem.Mol): An RDKit Mol object.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A tuple of molecular graph descriptors (node features, adjacency matrix, distance matrix).
    """
    node_features = np.array([get_atom_features(atom, one_hot_formal_charge) for atom in mol.GetAtoms()])
    # print('debug node features {}'.format(node_features.shape))

    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

    conf = mol.GetConformer()
    pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                           for k in range(mol.GetNumAtoms())])
    dist_matrix = pairwise_distances(pos_matrix)

    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

        m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_matrix
        dist_matrix = m

    return node_features, adj_matrix, dist_matrix


def get_atom_features(atom, one_hot_formal_charge=True):
    """Calculate atom features.

    Args:
        atom (rdchem.Atom): An RDKit Atom object.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A 1-dimensional array (ndarray) of atom features.
    """
    attributes = []

    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )

    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5]
    )

    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4]
    )

    if one_hot_formal_charge:
        attributes += one_hot_vector(
            atom.GetFormalCharge(),
            [-1, 0, 1]
        )
    else:
        attributes.append(atom.GetFormalCharge())

    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32)


def get_spurious_atom_features(neighbors=0, add_dummy_node=True, one_hot_formal_charge=True):
    """Calculate features for fake atoms.
        Args:
            neighbors (int): neighbors of dummy node
            add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
            one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.
        Returns:
            A 1-dimensional array (ndarray) of atom features.
        """
    attributes = []
    if add_dummy_node:
        atom_types = [-999, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]  # -999 used to denote dummy node type
    else:
        atom_types = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    attributes += one_hot_vector(999, atom_types)  # atom number
    attributes += one_hot_vector(neighbors, [0, 1, 2, 3, 4, 5])  # Neighbors
    attributes += one_hot_vector(0, [0, 1, 2, 3, 4])  # TotalNumHs
    if one_hot_formal_charge:
        attributes += one_hot_vector(0, [-1, 0, 1])  # FormalCharge
    else:
        attributes.append(0)
    attributes.append(False)  # IsInRing
    attributes.append(False)  # IsAromatic

    return np.array(attributes, dtype=np.float32)


def get_cat_matrix(m1, m2, pad_value):
    """ prepare matrix feature by cat two matrix """
    x1_len, x2_len = m1.shape[0], m2.shape[0]
    cat_len = x1_len + x2_len
    cat_matrix = np.full((cat_len, cat_len), pad_value)
    cat_matrix[0: x1_len, 0: x1_len] = m1
    cat_matrix[x1_len: cat_len, x1_len: cat_len] = m2
    return cat_matrix


def get_rn_matrix(m1, m2, rn_value, rn_self_value, pad_value):
    """ cat feature matrix and extend values of for relation node at the first row and column  """
    cat_matrix = get_cat_matrix(m1, m2, pad_value)
    m = np.full((cat_matrix.shape[0] + 1, cat_matrix.shape[1] + 1), rn_value)
    m[1:, 1:] = cat_matrix
    m[0, 0] = rn_self_value
    return m


def build_joint_features(x1_all, x2_all, add_dummy_node=True, one_hot_formal_charge=True, add_relation_node=False,
                         rn_dst=1, cross_dst=1e6):
    joint_features = []
    for x1, x2 in zip(x1_all, x2_all):
        node_ft1, adj_m1, dst_m1, smile1 = x1
        node_ft2, adj_m2, dst_m2, smile2 = x2
        cat_len = node_ft1.shape[0] + node_ft2.shape[0]

        cross_adj = 1 if cross_dst < 1e4 else 0  # adjacency between two mol
        if add_relation_node:
            rn_features = get_spurious_atom_features(cat_len, add_dummy_node, one_hot_formal_charge)
            node_features = np.concatenate(([rn_features], node_ft1, node_ft2), axis=0)
            adjacency_matrix = get_rn_matrix(adj_m1, adj_m2, rn_value=1, rn_self_value=1, pad_value=cross_adj)
            distance_matrix = get_rn_matrix(dst_m1, dst_m2, rn_value=rn_dst, rn_self_value=0, pad_value=cross_dst)
        else:
            node_features = np.concatenate((node_ft1, node_ft2), axis=0)
            adjacency_matrix = get_cat_matrix(adj_m1, adj_m2, pad_value=cross_adj)
            distance_matrix = get_cat_matrix(dst_m1, dst_m2, pad_value=cross_dst)

        # np.set_printoptions(precision=2)
        # print("\n ~~~~~~~~~~\ndst1 {} {}\n dst2 {} {}\n j dst {} {}\n adj1 {} {}\n adj2 {} {}\n j adj {} {}\n".format(
        #     dst_m1.shape, dst_m1, dst_m2.shape, dst_m2, distance_matrix.shape, distance_matrix,
        #     adj_m1.shape, adj_m1, adj_m2.shape, adj_m2, adjacency_matrix.shape, adjacency_matrix, ))

        joint_features.append([node_features, adjacency_matrix, distance_matrix])
    return joint_features


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


class Sample:
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """

    def __init__(self, x1, x2, joint, y, index):
        self.node_features1 = x1[0]
        self.adjacency_matrix1 = x1[1]
        self.distance_matrix1 = x1[2]
        self.smile1 = x1[3]

        self.node_features2 = x2[0]
        self.adjacency_matrix2 = x2[1]
        self.distance_matrix2 = x2[2]
        self.smile2 = x2[3]

        self.joint_node_features = joint[0]
        self.joint_adjacency_matrix = joint[1]
        self.joint_distance_matrix = joint[2]

        self.y = y
        self.index = index


class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of Molecule objects
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MolDataset(self.data_list[key])
        return self.data_list[key]


def pad_array(array, shape, dtype=np.float32):
    """Pad a 2-dimensional array with zeros.

    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


def fake_mol_collate_func(batch):
    """Create a padded batch of molecule features.

    Args:
        batch (list[Sample]): A batch of raw samples.

    Returns:
        A list of FloatTensors with padded molecule features:
         node features, mask, adjacency matrices, distance matrices, and labels.
    """
    adj_lst1, dist_lst1, emb_lst1, mask1 = [], [], [], []
    adj_lst2, dist_lst2, emb_lst2, mask2 = [], [], [], []
    labels = []

    max_size1 = 0
    max_size2 = 0

    for sample in batch:
        if random.random() > 0.5:  # construct fake data:
            sample.node_features1 = np.array([[1.0, 1.0]])
            sample.node_features2 = np.array([[1.0, 1.0]])
            labels.append(1.0)
        else:
            sample.node_features1 = np.array([[1.0, -1.0]])
            sample.node_features2 = np.array([[-1.0, 1.0]])
            labels.append(-1.0)

        if sample.adjacency_matrix1.shape[0] > max_size1:
            max_size1 = sample.adjacency_matrix1.shape[0]
        if sample.adjacency_matrix2.shape[0] > max_size2:
            max_size2 = sample.adjacency_matrix2.shape[0]

    for sample in batch:
        x1_len, x2_len = sample.node_features1.shape[0], sample.node_features2.shape[0]
        adj_lst1.append(pad_array(sample.adjacency_matrix1, (max_size1, max_size1)))
        dist_lst1.append(pad_array(sample.distance_matrix1, (max_size1, max_size1)))
        emb_lst1.append(sample.node_features1)
        mask1.append(np.concatenate((np.ones(x1_len, dtype=bool), np.zeros(max_size1 - x1_len, dtype=bool)), axis=0))

        adj_lst2.append(pad_array(sample.adjacency_matrix2, (max_size2, max_size2)))
        dist_lst2.append(pad_array(sample.distance_matrix2, (max_size2, max_size2)))
        emb_lst2.append(sample.node_features2)
        mask2.append(np.concatenate((np.ones(x2_len, dtype=bool), np.zeros(max_size2 - x2_len, dtype=bool)), axis=0))

    # build tensor and send to device
    features = (emb_lst1, adj_lst1, dist_lst1, emb_lst2, adj_lst2, dist_lst2, labels)
    emb1, adj1, dist1, emb2, adj2, dist2, labels = [FloatTensor(f) for f in features]
    mask1, mask2 = BoolTensor(mask1), BoolTensor(mask2)
    return emb1, mask1, adj1, dist1, emb2, mask2, adj2, dist2, labels


def mol_collate_func(batch):
    """Create a padded batch of molecule features.

    Args:
        batch (list[Sample]): A batch of raw samples.

    Returns:
        A list of FloatTensors with padded molecule features:
         node features, mask, adjacency matrices, distance matrices, and labels.
    """
    adj_lst1, dist_lst1, emb_lst1, mask1 = [], [], [], []
    adj_lst2, dist_lst2, emb_lst2, mask2 = [], [], [], []
    labels = []

    max_size1 = 0
    max_size2 = 0
    for sample in batch:
        labels.append(sample.y)
        if sample.adjacency_matrix1.shape[0] > max_size1:
            max_size1 = sample.adjacency_matrix1.shape[0]
        if sample.adjacency_matrix2.shape[0] > max_size2:
            max_size2 = sample.adjacency_matrix2.shape[0]

    for sample in batch:
        x1_len, x2_len = sample.node_features1.shape[0], sample.node_features2.shape[0]
        adj_lst1.append(pad_array(sample.adjacency_matrix1, (max_size1, max_size1)))
        dist_lst1.append(pad_array(sample.distance_matrix1, (max_size1, max_size1)))
        emb_lst1.append(pad_array(sample.node_features1, (max_size1, sample.node_features1.shape[1])))
        mask1.append(np.concatenate((np.ones(x1_len, dtype=bool), np.zeros(max_size1 - x1_len, dtype=bool)), axis=0))

        adj_lst2.append(pad_array(sample.adjacency_matrix2, (max_size2, max_size2)))
        dist_lst2.append(pad_array(sample.distance_matrix2, (max_size2, max_size2)))
        emb_lst2.append(pad_array(sample.node_features2, (max_size2, sample.node_features2.shape[1])))
        mask2.append(np.concatenate((np.ones(x2_len, dtype=bool), np.zeros(max_size2 - x2_len, dtype=bool)), axis=0))
    # build tensor and send to device
    features = (emb_lst1, adj_lst1, dist_lst1, emb_lst2, adj_lst2, dist_lst2, labels)
    emb1, adj1, dist1, emb2, adj2, dist2, labels = [FloatTensor(f) for f in features]
    mask1, mask2 = BoolTensor(mask1), BoolTensor(mask2)
    return emb1, mask1, adj1, dist1, emb2, mask2, adj2, dist2, labels


def joint_collate_func(batch):
    """
    Create a padded batch of molecule features with jointed feature (and relation node).
    Notice: different from mol collate func, it concatenate two mol graph into a single graph (with a relation node).

    Args:
        batch (list[Sample]): A batch of raw samples.

    Returns:
        A list of FloatTensors with padded molecule features:
         node features, mask, adjacency matrices, distance matrices, and labels.
    """
    adj_lst, dist_lst, emb_lst, mask = [], [], [], []
    labels = []

    max_len = 0
    for sample in batch:
        labels.append(sample.y)
        if sample.joint_adjacency_matrix.shape[0] > max_len:
            max_len = sample.joint_adjacency_matrix.shape[0]

    for sample in batch:
        joint_len = sample.joint_node_features.shape[0]
        adj_lst.append(pad_array(sample.joint_adjacency_matrix, (max_len, max_len)))
        dist_lst.append(pad_array(sample.joint_distance_matrix, (max_len, max_len)))
        emb_lst.append(pad_array(sample.joint_node_features, (max_len, sample.joint_node_features.shape[1])))
        mask.append(np.concatenate((np.ones(joint_len, dtype=bool), np.zeros(max_len - joint_len, dtype=bool)), axis=0))
        # print("debug padding:  ============\n dst {} {} \n pad dst {} {}\n adj {} {} \n pad adj {} {} \n mask {} {}".format(
        #     sample.joint_distance_matrix.shape, sample.joint_distance_matrix,
        #     pad_array(sample.joint_distance_matrix, (max_len, max_len)).shape, pad_array(sample.joint_distance_matrix, (max_len, max_len)),
        #     sample.joint_adjacency_matrix.shape, sample.joint_adjacency_matrix,
        #     pad_array(sample.joint_adjacency_matrix, (max_len, max_len)).shape, pad_array(sample.joint_adjacency_matrix, (max_len, max_len)),
        #     np.concatenate((np.ones(joint_len, dtype=bool), np.zeros(max_len - joint_len, dtype=bool)), axis=0).shape, np.concatenate((np.ones(joint_len, dtype=bool), np.zeros(max_len - joint_len, dtype=bool)), axis=0)
        # ))

    # build tensor and send to device
    emb, adj, dist,  labels = [FloatTensor(f) for f in (emb_lst, adj_lst, dist_lst, labels)]
    mask = BoolTensor(mask)
    return emb, mask, adj, dist, labels


def construct_dataset(x1_all, x2_all, joint_all, y_all):
    """Construct a MolDataset object from the provided data.

    Args:
        x1_all (list): A list of molecule features.
        x2_all (list): A list of molecule features.
        joint_all (list): A list of joint feature for x1 and x2.
        y_all (list): A list of the corresponding labels.

    Returns:
        A MolDataset object filled with the provided data.
    """
    output = [Sample(x1=data[0], x2=data[1], joint=data[2], y=data[3], index=i)
              for i, data in enumerate(zip(x1_all, x2_all, joint_all, y_all))]
    return MolDataset(output)


def construct_loader(x1, x2, joint, y, batch_size, shuffle=True, joint_enc=False):
    """Construct a data loader for the provided data.

    Args:
        x1 (list): A list of molecule features.
        x2 (list): A list of molecule features.
        joint (list): A list of joint feature for x1 and x2.
        y (list): A list of the corresponding labels.
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.
        joint_enc (bool): If True each batch will be concatenated mol features, false for separate features.

    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    data_set = construct_dataset(x1, x2, joint, y)
    collate_fn = joint_collate_func if joint_enc else mol_collate_func
    loader = torch.utils.data.DataLoader(
        dataset=data_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    return loader
