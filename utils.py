import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch.fft
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter
import pickle

np.set_printoptions(precision=4)


def set_gpu(gpus):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param

def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def cconv(a, b):
    return torch.fft.irfft(com_mult(torch.fft.rfft(
        a, 1), torch.fft.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def ccorr(a, b):
    return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a, 1)), torch.fft.rfft(
        b, 1)), 1, signal_sizes=(a.shape[-1],))


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def load_data(file_path):
    print("load data from {}".format(file_path))
    try:
        with open(os.path.join(file_path, 'ent2id.pkl'), 'rb') as f:
            entity2id = pickle.load(f)
    except BaseException:
        with open(os.path.join(file_path, 'entities.dict')) as f:
            entity2id = dict()

            for line in f:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)
    try:
        with open(os.path.join(file_path, 'rel2id.pkl'), 'rb') as f:
            relation2id = pickle.load(f)
    except BaseException:
        with open(os.path.join(file_path, 'relations.dict')) as f:
            relation2id = dict()

            for line in f:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

    train_triplets = read_triplets(
        os.path.join(
            file_path,
            'train.txt'),
        entity2id,
        relation2id)
    valid_triplets = read_triplets(
        os.path.join(
            file_path,
            'valid.txt'),
        entity2id,
        relation2id)
    test_triplets = read_triplets(
        os.path.join(
            file_path,
            'test.txt'),
        entity2id,
        relation2id)

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))

    return entity2id, relation2id, train_triplets, valid_triplets, test_triplets


def read_triplets(file_path, entity2id, relation2id):
    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append(
                (entity2id[head],
                 relation2id[relation],
                 entity2id[tail]))

    return np.array(triplets)


def sample_edge_uniform(n_triples, sample_size):
    all_edges = np.arange(n_triples)
    return np.random.choice(all_edges, sample_size, replace=False)


def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels


def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    one_hot = F.one_hot(
        edge_type.clone().detach().to(
            torch.long),
        num_classes=2 *
        num_relation).to(
            torch.float)
    deg = scatter_add(
        one_hot,
        edge_index[0].clone().detach().to(
            torch.long),
        dim=0,
        dim_size=num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / \
        deg[edge_index[0].clone().detach().to(torch.long)].view(-1)[index]

    return edge_norm


def generate_sampled_graph_and_labels(
        triplets,
        sample_size,
        split_size,
        num_entity,
        num_rels,
        negative_rate):

    edges = sample_edge_uniform(len(triplets), sample_size)

    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    samples, labels = negative_sampling(
        relabeled_edges, len(uniq_entity), negative_rate)

    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)

    src = torch.tensor(src[graph_split_ids], dtype=torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype=torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype=torch.long).contiguous()

    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(
        edge_type, edge_index, len(uniq_entity), num_rels)
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    return data


def build_test_graph(num_nodes, num_rels, triplets):
    src, rel, dst = triplets.transpose()

    src = torch.from_numpy(src)
    rel = torch.from_numpy(rel)
    dst = torch.from_numpy(dst)

    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(np.arange(num_nodes))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(
        edge_type, edge_index, num_nodes, num_rels)

    return data