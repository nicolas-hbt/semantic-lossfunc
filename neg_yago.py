import torch
import random
import numpy as np
import time
from collections import defaultdict
import pandas as pd
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

neg_nb = 1000

def antitc_except(triple, side, num_ent):
    corr = random.randint(0, num_ent - 1)
    if side == 'head':
        while r2id2dom2id[triple[1]] in instype_all[corr]:
            corr = random.randint(0, num_ent - 1)
    else:
        while r2id2range2id[triple[1]] in instype_all[corr]:
            corr = random.randint(0, num_ent - 1)
    return int(corr)


def get_observed_triples(train2id, valid2id, test2id):
    all_possible_hs = defaultdict(dict)
    all_possible_ts = defaultdict(dict)
    train2id = train2id[(train2id['r'].isin(r2id2dom2id.keys())) & (
        train2id['r'].isin(r2id2range2id.keys()))]
    train2id = train2id[~train2id['r'].isin(rels_suppr)]
    train2id = torch.as_tensor(train2id.to_numpy(), dtype=torch.int32)
    valid2id = torch.as_tensor(valid2id.to_numpy(), dtype=torch.int32)
    test2id = torch.as_tensor(test2id.to_numpy(), dtype=torch.int32)
    all_triples = torch.cat((train2id, valid2id, test2id))
    X = all_triples.detach().clone()
    for triple in range(X.shape[0]):
        h, r, t = X[triple][0].item(), X[triple][1].item(), X[triple][2].item()
        try:
            all_possible_ts[h][r].append(t)
        except KeyError:
            all_possible_ts[h][r] = [t]

    for triple in range(X.shape[0]):
        h, r, t = X[triple][0].item(), X[triple][1].item(), X[triple][2].item()
        try:
            all_possible_hs[t][r].append(h)
        except KeyError:
            all_possible_hs[t][r] = [h]

    all_possible_ts = dict(all_possible_ts)
    all_possible_hs = dict(all_possible_hs)
    return all_possible_hs, all_possible_ts


def sem_neg_files(train2id, neg_nb):
    start = time.time()
    sem_hr_, sem_tr_ = defaultdict(dict), defaultdict(dict)
    train2id = train2id.to_numpy()
    for idx, triple in enumerate(train2id):
        h, r, t = triple[0], triple[1], triple[2]
        if (len(class2id2ent2id[r2id2range2id[r]]) > 1) and (
                len(class2id2ent2id[r2id2dom2id[r]]) > 1):
            if not (h in sem_hr_ and r in sem_hr_[h]):
                sem_t = list(set(np.random.choice(
                    class2id2ent2id[r2id2range2id[r]], size=neg_nb)))
                sem_hr_[h][r] = sem_t

            if not (t in sem_tr_ and r in sem_tr_[t]):
                sem_h = list(set(np.random.choice(
                    class2id2ent2id[r2id2dom2id[r]], size=neg_nb)))
                sem_tr_[t][r] = sem_h

        if idx % 50000 == 0:
            print(idx, ' triples processed.')

    print('total time:', time.time() - start)
    sem_hr_, sem_tr_ = dict(sem_hr_), dict(sem_tr_)

    start = time.time()
    print('Filtering.')
    for h, rts in sem_hr_.items():
        for r, ts in rts.items():
            intersect = set(ts).intersection(all_possible_ts[h][r])
            if len(intersect) > 0:
                sem_hr_[h][r] = list(set(ts) - set(all_possible_ts[h][r]))

    for t, rhs in sem_tr_.items():
        for r, hs in rhs.items():
            intersect = set(hs).intersection(all_possible_hs[t][r])
            if len(intersect) > 0:
                sem_tr_[t][r] = list(set(hs) - set(all_possible_hs[t][r]))
    print('total time:', time.time() - start)
    return sem_hr_, sem_tr_


def dumb_neg_files(train2id, neg_nb):
    start = time.time()
    dumb_hr_, dumb_tr_ = defaultdict(dict), defaultdict(dict)
    train2id = train2id[(train2id['r'].isin(r2id2dom2id.keys())) & (
        train2id['r'].isin(r2id2range2id.keys()))]
    train2id = train2id.to_numpy()
    for idx, triple in enumerate(train2id):
        h, r, t = triple[0], triple[1], triple[2]
        dumb_hr_[h][r], dumb_tr_[t][r] = [], []

        for i in range(neg_nb):
            dumb_t = antitc_except(triple, side='tail', num_ent=len(ent2id))
            dumb_hr_[h][r].append(dumb_t)

            dumb_h = antitc_except(triple, side='head', num_ent=len(ent2id))
            dumb_tr_[t][r].append(dumb_h)

        dumb_hr_[h][r] = list(set(dumb_hr_[h][r]))
        dumb_tr_[t][r] = list(set(dumb_tr_[t][r]))
        if idx % 50000 == 0:
            print(idx, ' triples processed.')

    print('total time:', time.time() - start)
    dumb_hr_, dumb_tr_ = dict(dumb_hr_), dict(dumb_tr_)
    start = time.time()

    print('Filtering.')
    for h, rts in dumb_hr_.items():
        for r, ts in rts.items():
            intersect = set(ts).intersection(all_possible_ts[h][r])
            if len(intersect) > 0:
                dumb_hr_[h][r] = list(set(ts) - set(all_possible_ts[h][r]))

    for t, rhs in dumb_tr_.items():
        for r, hs in rhs.items():
            intersect = set(hs).intersection(all_possible_hs[t][r])
            if len(intersect) > 0:
                dumb_tr_[t][r] = list(set(hs) - set(all_possible_hs[t][r]))
    print('total time:', time.time() - start)

    return dict(dumb_hr_), dict(dumb_tr_)

dataset = 'datasets/YAGO14k/'

train2id = pd.read_csv(
    dataset +
    "train2id.txt",
    sep='\t',
    header=None,
    names=[
        'h',
        'r',
        't'])
valid2id = pd.read_csv(
    dataset +
    "valid2id.txt",
    sep='\t',
    header=None,
    names=[
        'h',
        'r',
        't'])
test2id = pd.read_csv(
    dataset +
    "test2id.txt",
    sep='\t',
    header=None,
    names=[
        'h',
        'r',
        't'])

with open(dataset + 'pickle/r2id2dom2id.pkl', 'rb') as f:
    r2id2dom2id = pickle.load(f)
with open(dataset + 'pickle/r2id2range2id.pkl', 'rb') as f:
    r2id2range2id = pickle.load(f)
with open(dataset + 'pickle/class2id2ent2id.pkl', 'rb') as f:
    class2id2ent2id = pickle.load(f)
with open(dataset + 'pickle/class2id.pkl', 'rb') as f:
    class2id = pickle.load(f)
with open(dataset + 'pickle/instype_all.pkl', 'rb') as f:
    instype_all = pickle.load(f)
with open(dataset + 'pickle/ent2id.pkl', 'rb') as f:
    ent2id = pickle.load(f)
with open(dataset + 'pickle/rel2id.pkl', 'rb') as f:
    rel2id = pickle.load(f)

train2id = train2id[(train2id['r'].isin(r2id2dom2id.keys()))
                    & (train2id['r'].isin(r2id2range2id.keys()))]
all_rels = train2id['r'].unique()
rels_suppr = []
pb_dom = list(set(r2id2dom2id.values()) - set(class2id2ent2id.keys()))
pb_range = list(set(r2id2range2id.values()) - set(class2id2ent2id.keys()))
for r in all_rels:
    if r2id2dom2id[r] in pb_dom:
        rels_suppr.append(r)
    if r2id2range2id[r] in pb_range:
        rels_suppr.append(r)

rels_suppr = list(set(rels_suppr))
train2id = train2id[~train2id['r'].isin(rels_suppr)]

all_possible_hs, all_possible_ts = get_observed_triples(
    train2id, valid2id, test2id)

print('sem negatives.')
neg_sem = 500

sem_hr_, sem_tr_ = sem_neg_files(train2id, neg_sem)
with open('datasets/Yago14k/pickle/sem_hr.pkl', 'wb') as f:
    pickle.dump(sem_hr_, f)
with open('datasets/Yago14k/pickle/sem_tr.pkl', 'wb') as f:
    pickle.dump(sem_tr_, f)

print('dumb negatives.')
neg_dumb = 500
dumb_hr_, dumb_tr_ = dumb_neg_files(train2id, neg_dumb)
with open('datasets/Yago14k/pickle/dumb_hr.pkl', 'wb') as f:
    pickle.dump(dumb_hr_, f)
with open('datasets/Yago14k/pickle/dumb_tr.pkl', 'wb') as f:
    pickle.dump(dumb_tr_, f)
