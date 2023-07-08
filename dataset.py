import os
import numpy as np
import pandas as pd
import pickle
import random
import torch
from numpy import genfromtxt
from collections import defaultdict
from sklearn.utils import shuffle
random.seed(7)
np.random.seed(7)


class Dataset:
    def __init__(self, ds_name, args):
        random.seed(7)
        np.random.seed(7)
        self.args = args
        self.device = args.device
        self.name = ds_name
        self.dir = "datasets/" + self.name + "/"
        self.ent2id, self.ent2id_typed, self.entid2typid = self.read_pickle(
            'ent2id'), self.read_pickle('ent2id_typed'), self.read_pickle('entid2typid')
        self.instype_spec, self.instype_transitive, self.instype_all = self.read_pickle(
            'instype_spec'), self.read_pickle('instype_transitive'), self.read_pickle('instype_all')
        self.rel2id = self.read_pickle('rel2id')
        self.class2id, self.class2id2ent2id, self.subclassof2id = self.read_pickle(
            'class2id'), self.read_pickle('class2id2ent2id'), self.read_pickle('subclassof2id')
        self.r2id2dom2id, self.r2id2range2id = self.read_pickle(
            'r2id2dom2id'), self.read_pickle('r2id2range2id')
        self.r2id2metadom2id, self.r2id2metarange2id = self.read_pickle(
            'r2id2metadom2id'), self.read_pickle('r2id2metarange2id')
        self.setting = args.setting
        self.sem = args.sem
        if self.args.lossfunc == 'bce':
            inv_r2id2dom2id = {}
            for k, v in self.r2id2dom2id.items():
                try:
                    inv_r2id2dom2id[k +
                                    max(self.rel2id.values()) +
                                    1] = self.r2id2range2id[k]
                except BaseException:
                    pass

            inv_r2id2range2id = {}
            for k, v in self.r2id2range2id.items():
                try:
                    inv_r2id2range2id[k +
                                      max(self.rel2id.values()) +
                                      1] = self.r2id2dom2id[k]
                except BaseException:
                    pass
            self.r2id2dom2id.update(inv_r2id2dom2id)
            self.r2id2range2id.update(inv_r2id2range2id)

        self.r2hs = self.read_pickle('heads2id_original')
        if self.args.lossfunc == 'bce':
            self.r2ts_origin = self.read_pickle('tails2id_original')
            self.r2ts = self.r2ts_origin.copy()
            for rel, tails in self.r2ts_origin.items():
                self.r2ts[rel + len(self.rel2id)] = self.r2hs[rel]
        else:
            self.r2ts = self.read_pickle('tails2id_original')

        self.data = {}

        if self.args.lossfunc == 'bce':
            self.data["pd_train"] = pd.read_csv(
                self.dir +
                "train2id_inv.txt",
                sep='\t',
                header=None,
                names=[
                    'h',
                    'r',
                    't'])
            self.data["pd_valid"] = pd.read_csv(
                self.dir +
                "valid2id_inv.txt",
                sep='\t',
                header=None,
                names=[
                    'h',
                    'r',
                    't'])
            self.data["pd_test"] = pd.read_csv(
                self.dir +
                "test2id_inv.txt",
                sep='\t',
                header=None,
                names=[
                    'h',
                    'r',
                    't'])
            self.data["train"] = self.data["pd_train"].to_numpy()
            self.data["valid"] = self.data["pd_valid"].to_numpy()
            self.data["test"] = self.data["pd_test"].to_numpy()
            self.data["df"] = pd.concat(
                [self.data["pd_train"], self.data["pd_valid"], self.data["pd_test"]]).to_numpy()

        else:
            self.data["pd_train"] = pd.read_csv(
                self.dir +
                "train2id.txt",
                sep='\t',
                header=None,
                names=[
                    'h',
                    'r',
                    't'])
            self.data["pd_valid"] = pd.read_csv(
                self.dir +
                "valid2id.txt",
                sep='\t',
                header=None,
                names=[
                    'h',
                    'r',
                    't'])
            self.data["pd_test"] = pd.read_csv(
                self.dir +
                "test2id.txt",
                sep='\t',
                header=None,
                names=[
                    'h',
                    'r',
                    't'])
            self.data["train"] = self.data["pd_train"].to_numpy()
            self.data["valid"] = self.data["pd_valid"].to_numpy()
            self.data["test"] = self.data["pd_test"].to_numpy()
            self.data["df"] = pd.concat(
                [self.data["pd_train"], self.data["pd_valid"], self.data["pd_test"]]).to_numpy()

        self.neg_ratio = args.neg_ratio
        self.neg_sampler = args.neg_sampler

        self.all_rels = list(self.rel2id.values())
        self.all_ents = list(self.ent2id.values())
        self.batch_index = 0

        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.id2class = {v: k for k, v in self.class2id.items()}

        self.dumb_hr_, self.dumb_tr_ = self.read_pickle(
            'dumb_hr'), self.read_pickle('dumb_tr')
        self.sem_hr_, self.sem_tr_ = self.read_pickle(
            'sem_hr'), self.read_pickle('sem_tr')

        all_possible_hs, all_possible_ts = self.get_observed_triples(
            self.data["train"], self.data["valid"], self.data["test"])
        if self.args.lossfunc == 'bce':
            with open(self.dir + 'pickle/observed_heads_inv.pkl', 'wb') as f:
                pickle.dump(all_possible_hs, f)
            with open(self.dir + 'pickle/observed_tails_inv.pkl', 'wb') as f:
                pickle.dump(all_possible_ts, f)
        else:
            with open(self.dir + 'pickle/observed_heads_original_kg.pkl', 'wb') as f:
                pickle.dump(all_possible_hs, f)
            with open(self.dir + 'pickle/observed_tails_original_kg.pkl', 'wb') as f:
                pickle.dump(all_possible_ts, f)

        self.data["train"] = shuffle(self.data["train"], random_state=7)

    def get_observed_triples(self, train2id, valid2id, test2id):
        all_possible_hs = defaultdict(dict)
        all_possible_ts = defaultdict(dict)
        train2id = torch.as_tensor(train2id, dtype=torch.int32)
        valid2id = torch.as_tensor(valid2id, dtype=torch.int32)
        test2id = torch.as_tensor(test2id, dtype=torch.int32)
        all_triples = torch.cat((train2id, valid2id, test2id))
        X = all_triples.detach().clone()
        for triple in range(X.shape[0]):
            h, r, t = X[triple][0].item(
            ), X[triple][1].item(), X[triple][2].item()
            try:
                all_possible_ts[h][r].append(t)
            except KeyError:
                all_possible_ts[h][r] = [t]

        for triple in range(X.shape[0]):
            h, r, t = X[triple][0].item(
            ), X[triple][1].item(), X[triple][2].item()
            try:
                all_possible_hs[t][r].append(h)
            except KeyError:
                all_possible_hs[t][r] = [h]

        all_possible_ts = dict(all_possible_ts)
        all_possible_hs = dict(all_possible_hs)
        return all_possible_hs, all_possible_ts

    def lookup_chunks(self):
        return len([filename for filename in os.listdir(
            self.dir + "pickle/") if filename.startswith("sem_tr_" + str(self.neg_sem))])

    def nb_neg(self):
        sem = [
            filename for filename in os.listdir(
                self.dir +
                "pickle/") if filename.startswith("sem_tr_")]
        dumb = [
            filename for filename in os.listdir(
                self.dir +
                "pickle/") if filename.startswith("dumb_tr_")]
        return int(sem[-1][7:8])

    def filtering(self):
        print('train shape before filtering: ', self.data["train"].shape[0])
        self.data["pd_train"] = self.data["pd_train"][(self.data["pd_train"]['r'].isin(
            self.r2id2dom2id.keys())) & (self.data["pd_train"]['r'].isin(self.r2id2range2id.keys()))]
        print(
            'train shape after filtering 0: ',
            self.data["pd_train"].shape[0])
        all_rels = self.data["pd_train"]['r'].unique()

        rels_suppr = []
        pb_dom = list(set(self.r2id2dom2id.values()) -
                      set(self.class2id2ent2id.keys()))
        pb_range = list(set(self.r2id2range2id.values()) -
                        set(self.class2id2ent2id.keys()))
        for r in all_rels:
            if self.r2id2dom2id[r] in pb_dom:
                rels_suppr.append(r)
            if self.r2id2range2id[r] in pb_range:
                rels_suppr.append(r)

        rels_suppr = list(set(rels_suppr))
        self.data["pd_train"] = self.data["pd_train"][~self.data["pd_train"]['r'].isin(
            rels_suppr)]
        print(
            'train shape after filtering 1: ',
            self.data["pd_train"].shape[0])

        if self.name != 'DBpedia77k':
            self.data["pd_train"] = self.data["pd_train"][((self.data["pd_train"]['h'].isin(
                self.sem_hr_.keys())) & (self.data["pd_train"]['t'].isin(self.sem_tr_.keys())))]
        print(
            'train shape after filtering 2: ',
            self.data["pd_train"].shape[0])
        self.data["train"] = self.data["pd_train"].to_numpy()

        idx_pb = []
        for i, triple in enumerate(self.data["train"]):
            h, r, t = triple[0], triple[1], triple[2]
            if self.name != 'FB15k187' and self.name != 'Yago14k':
                for i in range(1, self.nb_chunks + 1):
                    if (h not in globals()[f'self.sem_hr_{self.neg_sem}_{i}']) or (
                            t not in globals()[f'self.sem_tr_{self.neg_sem}_{i}']):
                        idx_pb.append(i)
                    elif (r not in globals()[f'self.sem_hr_{self.neg_sem}_{i}'][h]) or (r not in globals()[f'self.sem_tr_{self.neg_sem}_{i}'][t]):
                        idx_pb.append(i)
            else:
                if (h not in self.sem_hr_) or (t not in self.sem_tr_):
                    idx_pb.append(i)
                if (r not in self.sem_hr_[h]) or (r not in self.sem_tr_[t]):
                    idx_pb.append(i)

        bad_df = self.data["pd_train"].index.isin(idx_pb)
        self.data["pd_train"] = self.data["pd_train"].loc[~bad_df]
        print('train shape after filtering: ', self.data["pd_train"].shape[0])

        keep_ents = list(set(list(self.data["pd_train"]['h'].unique(
        )) + list(self.data["pd_train"]['t'].unique())))
        keep_rels = list(self.data["pd_train"]['r'].unique())

        print('valid shape before filtering: ', self.data["pd_valid"].shape[0])
        self.data["pd_valid"] = self.data["pd_valid"][self.data["pd_valid"]['r'].isin(
            keep_rels) & self.data["pd_valid"]['h'].isin(keep_ents) & self.data["pd_valid"]['t'].isin(keep_ents)]
        print('valid shape after filtering: ', self.data["pd_valid"].shape[0])

        print('test shape before filtering: ', self.data["pd_test"].shape[0])
        self.data["pd_test"] = self.data["pd_test"][self.data["pd_test"]['r'].isin(
            keep_rels) & self.data["pd_test"]['h'].isin(keep_ents) & self.data["pd_test"]['t'].isin(keep_ents)]
        print('test shape after filtering: ', self.data["pd_test"].shape[0])

        print('total entities: ', len(keep_ents))
        print('total relations: ', len(keep_rels))

        return self.data["pd_train"].to_numpy(
        ), self.data["pd_valid"].to_numpy(), self.data["pd_test"].to_numpy()

    def read_pickle(self, file):
        try:
            with open(self.dir + "pickle/" + file + ".pkl", 'rb') as f:
                pckl = pickle.load(f)
                return pckl
        except BaseException:
            print(file + ".pkl not found.")

    def num_ent(self):
        return len(self.ent2id)

    def num_class(self):
        return len(self.class2id)

    def num_rel(self):
        return len(self.rel2id)
    
    def was_last_batch(self):
        return (self.batch_index == 0)

    def next_pos_batch(self, batch_size):
        if self.batch_index + batch_size < len(self.data["train"]):
            batch = self.data["train"][self.batch_index: self.batch_index + batch_size]
            self.batch_index += batch_size
        else:
            batch = self.data["train"][self.batch_index:]
            self.batch_index = 0
        np.random.shuffle(batch)
        return batch

    def sem_picking(self, neg_batch, side):
        idx_tails = np.where(side == 2)[0].tolist()
        idx_heads = np.where(side == 0)[0].tolist()
        for i, triple in enumerate(neg_batch[np.where(side == 2)]):
            try:
                neg_batch[idx_tails[i], 2] = np.random.choice(
                    self.sem_hr_[triple[0]][triple[1]])
            except ValueError:
                neg_batch[idx_tails[i], 0] = np.random.choice(
                    self.sem_tr_[triple[2]][triple[1]])
            except KeyError:
                neg_batch[idx_tails[i], 2] = np.random.choice(
                    self.dumb_hr_[triple[0]][triple[1]])
        for i, triple in enumerate(neg_batch[np.where(side == 0)]):
            try:
                neg_batch[idx_heads[i], 0] = np.random.choice(
                    self.sem_tr_[triple[2]][triple[1]])
            except ValueError:
                neg_batch[idx_heads[i], 2] = np.random.choice(
                    self.sem_hr_[triple[0]][triple[1]])
            except KeyError:
                neg_batch[idx_heads[i], 0] = np.random.choice(
                    self.dumb_tr_[triple[2]][triple[1]])

        return neg_batch

    def dumb_picking(self, neg_batch, side):
        idx_tails = np.where(side == 2)[0].tolist()
        idx_heads = np.where(side == 0)[0].tolist()
        for i, triple in enumerate(neg_batch[np.where(side == 2)]):
            neg_batch[idx_tails[i], 2] = np.random.choice(
                self.dumb_hr_[triple[0]][triple[1]])
        for i, triple in enumerate(neg_batch[np.where(side == 0)]):
            neg_batch[idx_heads[i], 0] = np.random.choice(
                self.dumb_tr_[triple[2]][triple[1]])
        return neg_batch

    def neg_picking(self, pos_batch, neg_ratio=1):
        neg_batch_sem = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        neg_batch_dumb = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        M = neg_batch_sem.shape[0]
        side = np.random.choice([0, 2], size=M)
        neg_batch_sem = self.sem_picking(neg_batch_sem, side)
        neg_batch_dumb = self.dumb_picking(neg_batch_dumb, side)
        return neg_batch_sem, neg_batch_dumb
