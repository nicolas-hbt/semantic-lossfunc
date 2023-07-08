import torch
from functools import reduce
from dataset import Dataset
import numpy as np
from numpy import genfromtxt
import time
from tqdm import tqdm
from torch.utils import data as torch_data
from utils import *
from models import *
from sklearn.utils import shuffle
from collections import defaultdict
import pickle
import logging
torch.manual_seed(7)
np.random.seed(7)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tester:
    def __init__(self, dataset, args, model_path, valid_or_test):
        torch.manual_seed(7)
        np.random.seed(7)
        self.args = args
        self.fast_testing = args.fast_testing
        self.hierarchy = args.hierarchy
        self.dataset = dataset
        self.name = args.dataset
        self.setting = args.setting
        self.device = args.device
        self.model_name = args.model
        self.lmbda = args.reg
        self.dir = "datasets/" + self.dataset.name + "/"
        if self.model_name == 'TransE':
            self.model = TransE(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device)
        if self.model_name == 'TransH':
            self.model = TransH(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device)
        if self.model_name == 'TransR':
            self.model = TransR(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device)
        if self.model_name == 'TransD':
            self.model = TransD(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device)
        if self.model_name == 'HAKE':
            self.model = HAKE(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device,
                args.margin)
        if self.model_name == 'RotatE':
            self.model = RotatE2(
                dataset.num_ent(),
                dataset.num_rel(),
                args,
                self.device,
                args.margin)
        if self.model_name == 'DistMult':
            self.model = DistMult(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device,
                args)
        if self.model_name == 'ComplEx':
            self.model = ComplEx(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device,
                args)
        if self.model_name == 'SimplE':
            self.model = SimplE(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device,
                self.lmbda,
                args)
        if self.model_name == 'TuckER':
            self.model = TuckER(
                dataset.num_ent(),
                dataset.num_rel(),
                args,
                self.device)
        if self.model_name == 'ConvE':
            self.model = ConvE(
                dataset.num_ent(),
                dataset.num_rel(),
                args,
                self.device)
        if self.model_name == 'ConvKB1D':
            self.model = ConvKB1D(
                dataset.num_ent(),
                dataset.num_rel(),
                args,
                self.device)
        if self.model_name == 'ConvKB2D':
            self.model = ConvKB2D(
                dataset.num_ent(),
                dataset.num_rel(),
                args,
                self.device)
        if self.model_name == 'RGCN':
            self.model = RGCN(
                dataset.num_ent(),
                dataset.num_rel(),
                args,
                self.device)
        if self.model_name == 'CompGCN':
            self.model = CompGCN_DistMult(
                dataset.num_ent(), dataset.num_rel(), args, self.device)

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.valid_or_test = valid_or_test
        self.batch_size = args.batch_size
        self.neg_ratio = args.neg_ratio
        self.neg_sampler = args.neg_sampler
        self.metric = args.metrics
        self.sem = args.sem

        if self.args.lossfunc == 'bce':
            with open(self.dir + "pickle/observed_tails_inv.pkl", 'rb') as f:
                self.all_possible_ts = pickle.load(f)
        else:
            with open(self.dir + "pickle/observed_tails_original_kg.pkl", 'rb') as f:
                self.all_possible_ts = pickle.load(f)
            with open(self.dir + "pickle/observed_heads_original_kg.pkl", 'rb') as f:
                self.all_possible_hs = pickle.load(f)

        def read_pickle(file):
            try:
                with open(self.dir + "pickle/" + file + ".pkl", 'rb') as f:
                    pckl = pickle.load(f)
                    return pckl
            except BaseException:
                print(file + ".pkl not found.")

        def count_rel():
            rel_counts_h = {}
            rel_counts_t = {}
            for k, v in self.dataset.r2id2dom2id.items():

                rel_counts_h[k] = len(
                    self.dataset.class2id2ent2id[self.dataset.r2id2dom2id[k]])
                rel_counts_t[k] = len(
                    self.dataset.class2id2ent2id[self.dataset.r2id2range2id[k]])

            rel_counts_h = dict(
                sorted(
                    rel_counts_h.items(),
                    key=lambda x: x[1]))
            rel_counts_t = dict(
                sorted(
                    rel_counts_t.items(),
                    key=lambda x: x[1]))
            return rel_counts_h, rel_counts_t

        self.count_hs, self.count_ts = count_rel()

        def create_buckets():
            buckets_h, buckets_t = {}, {}
            if self.name == 'Yago14k':
                for r, hs in self.count_hs.items():
                    if hs < 2102:
                        buckets_h[r] = 1
                    elif hs <= 3624:
                        buckets_h[r] = 2
                    else:
                        buckets_h[r] = 3
                for r, ts in self.count_ts.items():
                    if ts < 2102:
                        buckets_t[r] = 1
                    elif ts <= 3624:
                        buckets_t[r] = 2
                    else:
                        buckets_t[r] = 3
            if self.name == 'FB15k187':
                for r, hs in self.count_hs.items():
                    if hs < 278:
                        buckets_h[r] = 1
                    elif hs <= 1391:
                        buckets_h[r] = 2
                    else:
                        buckets_h[r] = 3
                for r, ts in self.count_ts.items():
                    if ts < 278:
                        buckets_t[r] = 1
                    elif ts <= 1391:
                        buckets_t[r] = 2
                    else:
                        buckets_t[r] = 3
            if self.name == 'DBpedia77k':
                for r, hs in self.count_hs.items():
                    if hs < 1295:
                        buckets_h[r] = 1
                    elif hs <= 11586:
                        buckets_h[r] = 2
                    else:
                        buckets_h[r] = 3
                for r, ts in self.count_ts.items():
                    if ts < 1419:
                        buckets_t[r] = 1
                    elif ts <= 11586:
                        buckets_t[r] = 2
                    else:
                        buckets_t[r] = 3

            return buckets_h, buckets_t

        self.buckets_h, self.buckets_t = create_buckets()

    def get_observed_triples(self, train2id, valid2id, test2id):
        all_possible_hs = defaultdict(dict)
        all_possible_ts = defaultdict(dict)
        train2id = torch.as_tensor(train2id.to_numpy(), dtype=torch.int32)
        valid2id = torch.as_tensor(valid2id.to_numpy(), dtype=torch.int32)
        test2id = torch.as_tensor(test2id.to_numpy(), dtype=torch.int32)
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

    def get_observed_h(self, h, r, t):
        return (
            list(set(self.all_possible_hs[t.item()][r.item()]) - set([h.item()])))

    def get_observed_t(self, h, r, t):
        try:
            return (
                list(set(self.all_possible_ts[h.item()][r.item()]) - set([t.item()])))
        except KeyError:
            return None

    def predictions(self, h, r, t, all_entities):
        heads = h.reshape(-1, 1).repeat(1, all_entities.size()
                                        [1]).to(self.device)
        rels = r.reshape(-1, 1).repeat(1, all_entities.size()
                                       [1]).to(self.device)
        tails = t.reshape(-1, 1).repeat(1, all_entities.size()
                                        [1]).to(self.device)
        triplets = torch.stack((heads, rels, all_entities),
                               dim=2).reshape(-1, 3).to(self.device)

        tails_predictions = self.model.forward2(
            (triplets[:, 0]), (triplets[:, 1]), (triplets[:, 2])).reshape(1, -1)
        triplets = torch.stack((all_entities, rels, tails),
                               dim=2).reshape(-1, 3).to(self.device)
        heads_predictions = self.model.forward2(
            (triplets[:, 0]), (triplets[:, 1]), (triplets[:, 2])).reshape(1, -1)
        return heads_predictions.squeeze(), tails_predictions.squeeze()

    def test(self):
        schema_CWA_h = {}
        schema_CWA_t = {}
        sem_h_triples_CWA = {}
        sem_t_triples_CWA = {}
        for b in range(1, self.args.buckets + 1):
            schema_CWA_h[b] = {'sem1_h': 0.0, 'sem3_h': 0.0, 'sem10_h': 0.0}
            schema_CWA_t[b] = {'sem1_t': 0.0, 'sem3_t': 0.0, 'sem10_t': 0.0}
            sem_h_triples_CWA[b] = 0
            sem_t_triples_CWA[b] = 0

        filt_hit_h, filt_hit_t = {}, {}
        for b in range(1, self.args.buckets + 1):
            filt_hit_h[b] = {'hit1_h': 0.0, 'hit3_h': 0.0, 'hit10_h': 0.0}
            filt_hit_t[b] = {'hit1_t': 0.0, 'hit3_t': 0.0, 'hit10_t': 0.0}

        filt_mrr, filt_mrr_h, filt_mrr_t = {}, {}, {}
        for b in range(1, self.args.buckets + 1):
            filt_mrr[b] = 0.0
            filt_mrr_h[b] = []
            filt_mrr_t[b] = []

        nb_buckets_h, nb_buckets_t = {}, {}
        for b in range(1, self.args.buckets + 1):
            nb_buckets_h[b] = 0
            nb_buckets_t[b] = 0

        zero_tensor = torch.tensor([0], device=self.device)
        one_tensor = torch.tensor([1], device=self.device)
        X_valid_or_test = torch.from_numpy(
            (self.dataset.data[self.valid_or_test]))
        num_ent = self.dataset.num_ent()
        all_entities = torch.arange(
            end=num_ent, device=self.device).unsqueeze(0)
        start = time.time()
        if self.args.lossfunc == 'bce':
            half_idx = int(X_valid_or_test.shape[0] / 2)
            X_tails = X_valid_or_test[:half_idx]
            X_inv = X_valid_or_test[half_idx:]
            for triple in tqdm(X_tails):
                h, r, t = triple[0], triple[1], triple[2]
                bucket = self.buckets_t[r.item()]
                nb_buckets_t[bucket] += 1
                rm_idx_t = self.get_observed_t(h, r, t)
                tails_predictions = self.model.get_score(
                    h.to(
                        self.device), r.to(
                        self.device), t.to(
                        self.device)).squeeze()
                tails_predictions[[rm_idx_t]] = - np.inf
                indices_tail = tails_predictions.argsort(descending=True)
                filt_rank_t = (
                    indices_tail == t).nonzero(
                    as_tuple=True)[0].item() + 1
                filt_mrr_t[bucket].append(1.0 / filt_rank_t)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2range2id.keys(
                        ) and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(
                                    self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both' or self.setting == 'CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(
                                        indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA_t[bucket]['sem1_t'], schema_CWA_t[bucket]['sem3_t'], schema_CWA_t[bucket]['sem10_t'] = schema_CWA_t[
                                        bucket]['sem1_t'] + s1, schema_CWA_t[bucket]['sem3_t'] + s3, schema_CWA_t[bucket]['sem10_t'] + s10
                                    sem_t_triples_CWA[bucket] += 1

                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit_t[bucket]['hit10_t'] += torch.where(
                        indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit_t[bucket]['hit3_t'] += torch.where(
                        indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit_t[bucket]['hit1_t'] += torch.where(
                        indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()

            # batch of reversed triples
            for triple in tqdm(X_inv):
                h, r, t = triple[0], triple[1], triple[2]
                bucket = self.buckets_h[r.item() - self.dataset.num_rel()]
                nb_buckets_h[bucket] += 1
                rm_idx_t = self.get_observed_t(h, r, t)
                tails_predictions = self.model.get_score(
                    h.to(
                        self.device), r.to(
                        self.device), t.to(
                        self.device)).squeeze()
                tails_predictions[[rm_idx_t]] = - np.inf
                indices_tail = tails_predictions.argsort(descending=True)
                filt_rank_h = (
                    indices_tail == t).nonzero(
                    as_tuple=True)[0].item() + 1
                filt_mrr_h[bucket].append(1.0 / filt_rank_h)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2range2id.keys(
                        ) and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(
                                    self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both' or self.setting == 'CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(
                                        indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA_h[bucket]['sem1_h'], schema_CWA_h[bucket]['sem3_h'], schema_CWA_h[bucket]['sem10_h'] = schema_CWA_h[
                                        bucket]['sem1_h'] + s1, schema_CWA_h[bucket]['sem3_h'] + s3, schema_CWA_h[bucket]['sem10_h'] + s10
                                    sem_h_triples_CWA[bucket] += 1

                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit_h[bucket]['hit10_h'] += torch.where(
                        indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit_h[bucket]['hit3_h'] += torch.where(
                        indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit_h[bucket]['hit1_h'] += torch.where(
                        indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()

        else:
            X_valid_or_test = torch.from_numpy(
                (self.dataset.data[self.valid_or_test]))
            for triple in tqdm(X_valid_or_test):
                h, r, t = triple[0], triple[1], triple[2]
                bucket_h = self.buckets_h[r.item()]
                bucket_t = self.buckets_t[r.item()]
                nb_buckets_h[bucket_h] += 1
                nb_buckets_t[bucket_t] += 1
                rm_idx_t = self.get_observed_t(h, r, t)
                rm_idx_h = self.get_observed_h(h, r, t)
                heads_predictions, tails_predictions = self.predictions(
                    h, r, t, all_entities)
                heads_predictions[[rm_idx_h]], tails_predictions[[
                    rm_idx_t]] = -np.inf, -np.inf
                indices_tail, indices_head = tails_predictions.argsort(
                    descending=True), heads_predictions.argsort(descending=True)
                filt_rank_h = (
                    indices_head == h).nonzero(
                    as_tuple=True)[0].item() + 1
                filt_rank_t = (
                    indices_tail == t).nonzero(
                    as_tuple=True)[0].item() + 1
                filt_mrr_h[bucket_h].append(1.0 / filt_rank_h)
                filt_mrr_t[bucket_t].append(1.0 / filt_rank_t)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2dom2id.keys(
                        ) and self.dataset.r2id2dom2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(
                                    self.dataset.class2id2ent2id[self.dataset.r2id2dom2id[r.item()]]) >= 10:
                                if self.setting == 'both' or self.setting == 'CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(
                                        indices_head[:100], r.item(), side='head', k=10, setting='CWA')
                                    schema_CWA_h[bucket_h]['sem1_h'], schema_CWA_h[bucket_h]['sem3_h'], schema_CWA_h[bucket_h]['sem10_h'] = schema_CWA_h[
                                        bucket_h]['sem1_h'] + s1, schema_CWA_h[bucket_h]['sem3_h'] + s3, schema_CWA_h[bucket_h]['sem10_h'] + s10
                                    sem_h_triples_CWA[bucket_h] += 1

                        if r.item() in self.dataset.r2id2range2id.keys(
                        ) and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(
                                    self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both' or self.setting == 'CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(
                                        indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA_t[bucket_t]['sem1_t'], schema_CWA_t[bucket_t]['sem3_t'], schema_CWA_t[bucket_t]['sem10_t'] = schema_CWA_t[
                                        bucket_t]['sem1_t'] + s1, schema_CWA_t[bucket_t]['sem3_t'] + s3, schema_CWA_t[bucket_t]['sem10_t'] + s10
                                    sem_t_triples_CWA[bucket_t] += 1

                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit_h[bucket_h]['hit10_h'] += torch.where(
                        indices_head[:10] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit_h[bucket_h]['hit3_h'] += torch.where(
                        indices_head[:3] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit_h[bucket_h]['hit1_h'] += torch.where(
                        indices_head[:1] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit_t[bucket_t]['hit10_t'] += torch.where(
                        indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit_t[bucket_t]['hit3_t'] += torch.where(
                        indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit_t[bucket_t]['hit1_t'] += torch.where(
                        indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()

        print(time.time() - start)
        for b in range(1, self.args.buckets + 1):
            filt_mrr_h[b] = np.mean(filt_mrr_t[b])
            filt_mrr_t[b] = np.mean(filt_mrr_h[b])
            filt_mrr[b] = (filt_mrr_h[b] + filt_mrr_t[b]) / 2

        filtered_hits_at_1 = {}
        filtered_hits_at_3 = {}
        filtered_hits_at_10 = {}
        for b in range(1, self.args.buckets + 1):
            filt_hit_h[b]['hit1_h'] /= nb_buckets_h[b]
            filt_hit_h[b]['hit3_h'] /= nb_buckets_h[b]
            filt_hit_h[b]['hit10_h'] /= nb_buckets_h[b]
            filt_hit_t[b]['hit1_t'] /= nb_buckets_t[b]
            filt_hit_t[b]['hit3_t'] /= nb_buckets_t[b]
            filt_hit_t[b]['hit10_t'] /= nb_buckets_t[b]
            filtered_hits_at_1[b] = (
                filt_hit_h[b]['hit1_h'] + filt_hit_t[b]['hit1_t']) / 2
            filtered_hits_at_3[b] = (
                filt_hit_h[b]['hit3_h'] + filt_hit_t[b]['hit3_t']) / 2
            filtered_hits_at_10[b] = (
                filt_hit_h[b]['hit10_h'] + filt_hit_t[b]['hit10_t']) / 2

        if (self.sem == 'schema' or self.sem ==
                'both') and self.metric != 'ranks':
            schema_CWA = {}
            for b in range(1, self.args.buckets + 1):
                schema_CWA_h[b]['sem1_h'] /= sem_h_triples_CWA[b]
                schema_CWA_t[b]['sem1_t'] /= sem_t_triples_CWA[b]
                schema_CWA_h[b]['sem3_h'] /= sem_h_triples_CWA[b]
                schema_CWA_t[b]['sem3_t'] /= sem_t_triples_CWA[b]
                schema_CWA_h[b]['sem10_h'] /= sem_h_triples_CWA[b]
                schema_CWA_t[b]['sem10_t'] /= sem_t_triples_CWA[b]
                schema_CWA[b] = {}
                schema_CWA[b]['sem1'] = (
                    schema_CWA_h[b]['sem1_h'] + schema_CWA_t[b]['sem1_t']) / 2
                schema_CWA[b]['sem3'] = (
                    schema_CWA_h[b]['sem3_h'] + schema_CWA_t[b]['sem3_t']) / 2
                schema_CWA[b]['sem10'] = (
                    schema_CWA_h[b]['sem10_h'] + schema_CWA_t[b]['sem10_t']) / 2

        if self.metric == 'sem' or self.metric == 'all':
            return filt_mrr_h, filt_mrr_t, filt_mrr, filt_hit_h, filt_hit_t, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_10, \
                schema_CWA_h, schema_CWA_t, schema_CWA

        else:
            return filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10,\
                filt_mrr_h, filt_mrr_t, filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t

    def sem_at_k(self, preds, rel, side='head', k=10, setting='CWA'):
        valid_types = []
        for pred in preds.tolist():
            if len(valid_types) == 10:
                return ((valid_types[0]),
                        np.mean(valid_types[:3]),
                        np.mean(valid_types[:5]),
                        np.mean(valid_types[:10]))
            else:
                try:
                    classes = self.dataset.instype_all[pred]
                    if side == 'head':
                        dom = self.dataset.r2id2dom2id[rel]
                        valid_types.append(1 if dom in classes else 0)
                    elif side == 'tail':
                        rang = self.dataset.r2id2range2id[rel]
                        valid_types.append(1 if rang in classes else 0)
                except KeyError:
                    valid_types.append(0)
