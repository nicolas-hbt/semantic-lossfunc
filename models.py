import math
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn.conv import MessagePassing
from utils import uniform
import numpy as np
from torch.nn.init import xavier_normal_, xavier_uniform_
torch.manual_seed(7)
np.random.seed(7)

### TRANSLATIONAL MODELS ###
class TransE(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device, margin=None):
        super(TransE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device
        self.distance = 'l2'
        self.epsilon = 2.0
        self.p_norm = 1
        torch.manual_seed(7)
        self.ent_embs = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_embs = nn.Embedding(num_rel, emb_dim).to(device)
        nn.init.xavier_uniform_(self.ent_embs.weight.data)
        nn.init.xavier_uniform_(self.rel_embs.weight.data)

    def forward2(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(
            self.device), self.rel_embs(rs).to(
            self.device), self.ent_embs(ts).to(
            self.device)
        score = self._calc(e_hs, e_rs, e_ts).view(-1, 1).to(self.device)
        return score

    def forward(self, batch):
        hs = torch.tensor(batch[:, 0]).long().to(self.device)
        rs = torch.tensor(batch[:, 1]).long().to(self.device)
        ts = torch.tensor(batch[:, 2]).long().to(self.device)
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(
            self.device), self.rel_embs(rs).to(
            self.device), self.ent_embs(ts).to(
            self.device)
        score = self._calc(e_hs, e_rs, e_ts).view(-1, 1).to(self.device)
        return score

    def _calc(self, e_hs, e_rs, e_ts):
        score = (e_hs + e_rs) - e_ts
        if self.distance == 'l1':
            score = torch.norm(score, self.p_norm, -1)
        else:
            score = torch.sqrt(torch.sum((score)**2, 1))
        return -score

    def _loss(self, y_pos, y_neg, neg_ratio, gamma):
        criterion = nn.MarginRankingLoss(margin=gamma)
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0] / P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        target = Variable(torch.from_numpy(
            np.ones(P * (int(y_neg.shape[0] / P)), dtype=np.int32))).to(self.device)
        loss = criterion(y_pos, y_neg, target)
        return loss

    def _softplus(self, scores, neg_ratio, beta, label):
        criterion = nn.Softplus(beta=beta)
        loss = torch.mean(criterion(-scores * label))
        return loss

    def _logistic_pairwise(self, y_pos, y_neg, neg_ratio, gamma=1.0):
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0] / P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        exp = torch.exp(y_neg - y_pos)
        loss = torch.mean(gamma * torch.log(1 + exp))
        return loss

    def forward_bce(self, e1, rel):
        self.emb_e(e1)
        self.emb_rel(rel)

        prediction = torch.sigmoid(output)

        return prediction

    def _bce_vanilla(self, pos_batch):
        preds = self.forward(pos_batch).to(self.device)
        preds = torch.sigmoid(preds)
        ts = torch.tensor(pos_batch[:, 2]).long().to(self.device)
        labels = self.get_batch(pos_batch.shape[0], ts).to(self.device)
        print('preds: ', preds, preds.shape)
        print('labels: ', labels, labels.shape)
        return nn.BCELoss(preds, labels)

    def _bce_sem(self, pos_batch, labels):
        preds = self.forward(pos_batch).to(self.device)
        return nn.BCELoss(preds, labels)

    def get_batch(self, batch_size, batch_t):
        targets = torch.zeros(batch_size, self.num_ent).scatter_(
            1, batch_t.cpu().view(-1, 1).type(torch.int64), 1).to(self.device)
        return targets

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(
            self.device), self.rel_embs(rs).to(
            self.device), self.ent_embs(ts).to(
            self.device)
        regul = (torch.mean(e_hs ** 2) + torch.mean(e_ts ** 2) +
                 torch.mean(e_rs ** 2)) / 3
        return regul


class TransH(nn.Module):
    def __init__(
            self,
            num_ent,
            num_rel,
            emb_dim,
            device,
            margin=2.0,
            norm_flag=True):
        super(TransH, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device
        self.distance = 'l2'
        self.gamma = margin
        self.margin = margin
        self.epsilon = 2.0
        self.norm = 1
        torch.manual_seed(7)
        self.ent_embs = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_embs = nn.Embedding(num_rel, emb_dim).to(device)
        self.norm_vector = nn.Embedding(
            self.num_rel, self.emb_dim).to(
            self.device)

        if margin is None or self.epsilon is None:
            nn.init.xavier_uniform_(self.ent_embs.weight.data)
            nn.init.xavier_uniform_(self.rel_embs.weight.data)
            nn.init.xavier_uniform_(self.norm_vector.weight.data)
        else:

            self.embedding_range = nn.Parameter(torch.Tensor(
                [(self.margin + self.epsilon) / self.emb_dim]).to(self.device), requires_grad=False)
            nn.init.uniform_(
                tensor=self.ent_embs.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embs.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.norm_vector.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
        if margin is not None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def forward2(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(
            self.device), self.rel_embs(rs).to(
            self.device), self.ent_embs(ts).to(
            self.device)
        n = self.norm_vector(rs).to(self.device)
        proj_e_hs, proj_e_ts = self._transfer(
            e_hs, n).to(
            self.device), self._transfer(
            e_ts, n).to(
                self.device)
        score = self._calc(proj_e_hs, e_rs, proj_e_ts).view(-1, 1)
        return score

    def forward(self, batch):
        hs = torch.tensor(batch[:, 0]).long().to(self.device)
        rs = torch.tensor(batch[:, 1]).long().to(self.device)
        ts = torch.tensor(batch[:, 2]).long().to(self.device)
        score = self.forward2(hs, rs, ts)
        return score

    def _transfer(self, embeddings, norm):
        return embeddings - torch.sum(embeddings * norm, 1, True) * norm

    def _calc(self, e_hs, e_rs, e_ts):
        score = (e_hs + e_rs) - e_ts
        if self.distance == 'l1':
            score = torch.norm(score, self.p_norm, -1)
        else:
            score = torch.sqrt(torch.sum((score)**2, 1))
        return -score

    def _loss(self, y_pos, y_neg, neg_ratio, gamma):
        criterion = nn.MarginRankingLoss(margin=gamma)
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0] / P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        target = Variable(torch.from_numpy(
            np.ones(P * (int(y_neg.shape[0] / P)), dtype=np.int32))).to(self.device)
        loss = criterion(y_pos, y_neg, target)
        return loss

    def _softplus(self, scores, neg_ratio, beta, label):
        criterion = nn.Softplus(beta=beta)
        loss = torch.mean(criterion(-scores * label))
        return loss

    def _logistic_pairwise(self, y_pos, y_neg, neg_ratio, gamma=1.0):
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0] / P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        exp = torch.exp(y_neg - y_pos)
        loss = torch.mean(gamma * torch.log(1 + exp))
        return loss

    def _bce_vanilla(self, pos_batch):
        preds = self.forward(pos_batch).to(self.device)
        ts = torch.tensor(pos_batch[:, 2]).long().to(self.device)
        labels = self.get_batch(pos_batch.shape[0], ts).to(self.device)
        return nn.BCELoss(preds, labels)

    def _bce_sem(self, pos_batch, labels):
        preds = self.forward(pos_batch).to(self.device)
        return nn.BCELoss(preds, labels)

    def get_batch(self, batch_size, batch_t):
        targets = torch.zeros(batch_size, self.num_ent).scatter_(
            1, batch_t.cpu().view(-1, 1).type(torch.int64), 1).to(self.device)
        return targets

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(
            self.device), self.rel_embs(rs).to(
            self.device), self.ent_embs(ts).to(
            self.device)
        r_norm = self.norm_vector(rs)
        regul = (torch.mean(e_hs ** 2) +
                 torch.mean(e_rs ** 2) +
                 torch.mean(e_ts ** 2) +
                 torch.mean(r_norm ** 2)) / 4
        return regul


### SEMANTIC-MATCHING MODELS ###
class DistMult(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device, args, margin=2.0):
        super(DistMult, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        if args.lossfunc == 'bce':
            self.num_rel = num_rel * 2
        self.emb_dim = emb_dim
        self.device = device
        self.gamma = margin
        self.BCE = nn.BCELoss()
        torch.manual_seed(7)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.ent_embs = nn.Embedding(self.num_ent, emb_dim).to(device)
        self.rel_embs = nn.Embedding(self.num_rel, emb_dim).to(device)
        nn.init.xavier_uniform_(self.ent_embs.weight.data)
        nn.init.xavier_uniform_(self.rel_embs.weight.data)

    def forward2(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(
            self.device), self.rel_embs(rs).to(
            self.device), self.ent_embs(ts).to(
            self.device)
        score = self._calc(e_hs, e_rs, e_ts)
        return score

    def forward(self, batch):
        hs = torch.tensor(batch[:, 0]).long().to(self.device)
        rs = torch.tensor(batch[:, 1]).long().to(self.device)
        ts = torch.tensor(batch[:, 2]).long().to(self.device)
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(
            self.device), self.rel_embs(rs).to(
            self.device), self.ent_embs(ts).to(
            self.device)
        score = self._calc(e_hs, e_rs, e_ts)
        return score

    def forward_bce(self, batch):
        e1 = torch.tensor(batch[:, 0]).long().to(self.device)
        rel = torch.tensor(batch[:, 1]).long().to(self.device)
        e1_embedded = self.ent_embs(e1)
        rel_embedded = self.rel_embs(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(
            e1_embedded *
            rel_embedded,
            self.ent_embs.weight.transpose(
                1,
                0))
        pred = torch.sigmoid(pred)

        return pred

    def get_score(self, h, r, t):
        e1_embedded = self.ent_embs(h)
        rel_embedded = self.rel_embs(r)
        pred = self.calc_bce(e1_embedded, rel_embedded)
        return pred

    def calc_bce(self, e1_embedded, rel_embedded):
        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)
        sp_ = (e1_embedded * rel_embedded).unsqueeze(0)
        pred = torch.mm(sp_, self.ent_embs.weight.transpose(1, 0))
        return pred

    def _calc(self, e_hs, e_rs, e_ts):
        return torch.sum(e_hs * e_rs * e_ts, -1)

    def _loss(self, y_pos, y_neg, neg_ratio, gamma):
        criterion = nn.MarginRankingLoss(margin=gamma)
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0] / P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        target = Variable(torch.from_numpy(
            np.ones(P * (int(y_neg.shape[0] / P)), dtype=np.int32))).to(self.device)
        loss = criterion(y_pos, y_neg, target)
        return loss

    def _softplus(self, scores, neg_ratio, beta, label):
        criterion = nn.Softplus(beta=beta)
        loss = torch.mean(criterion(-scores * label))
        return loss

    def _logistic_pairwise(self, y_pos, y_neg, neg_ratio, gamma=1.0):
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0] / P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        exp = torch.exp(y_neg - y_pos)
        loss = torch.mean(gamma * torch.log(1 + exp))
        return loss

    def _bce_vanilla(self, pos_batch):
        preds = self.forward_bce(pos_batch).to(self.device)
        ts = torch.tensor(pos_batch[:, 2]).long().to(self.device)
        labels = self.get_batch(pos_batch.shape[0], ts).to(self.device)
        return self.BCE(preds, labels)

    def _bce_sem(self, pos_batch, labels):
        preds = self.forward_bce(pos_batch).to(self.device)
        return self.BCE(preds, labels)

    def get_batch(self, batch_size, batch_t):
        targets = torch.zeros(batch_size, self.num_ent).scatter_(
            1, batch_t.cpu().view(-1, 1).type(torch.int64), 1).to(self.device)
        return targets

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(
            self.device), self.rel_embs(rs).to(
            self.device), self.ent_embs(ts).to(
            self.device)
        regul = (torch.mean(e_hs ** 2) + torch.mean(e_ts ** 2) +
                 torch.mean(e_rs ** 2)) / 3
        return regul


class SimplE(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device, lmbda, args):
        super(SimplE, self).__init__()
        self.dim = emb_dim
        self.num_ent = num_ent
        self.num_rel = num_rel
        if args.lossfunc == 'bce':
            self.num_rel = num_rel * 2
        self.device = device
        self.lmbda = lmbda
        self.BCE = nn.BCELoss()
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        torch.manual_seed(7)
        self.ent_h_embs = nn.Embedding(self.num_ent, self.dim).to(self.device)
        self.ent_t_embs = nn.Embedding(self.num_ent, self.dim).to(self.device)
        self.rel_embs = nn.Embedding(self.num_rel, self.dim).to(self.device)
        self.rel_inv_embs = nn.Embedding(
            self.num_rel, self.dim).to(
            self.device)
        sqrt_size = 6.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.ent_h_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.ent_t_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_inv_embs.weight.data, -sqrt_size, sqrt_size)

    def forward2(self, heads, rels, tails):
        hh_embs = self.ent_h_embs(heads).to(self.device)
        ht_embs = self.ent_h_embs(tails).to(self.device)
        th_embs = self.ent_t_embs(heads).to(self.device)
        tt_embs = self.ent_t_embs(tails).to(self.device)
        r_embs = self.rel_embs(rels).to(self.device)
        r_inv_embs = self.rel_inv_embs(rels).to(self.device)

        scores1 = torch.sum(hh_embs * r_embs * tt_embs, dim=1)
        scores2 = torch.sum(ht_embs * r_inv_embs * th_embs, dim=1)
        return torch.clamp((scores1 + scores2) / 2, -20, 20)

    def forward(self, batch):
        heads = torch.tensor(batch[:, 0]).long().to(self.device)
        rels = torch.tensor(batch[:, 1]).long().to(self.device)
        tails = torch.tensor(batch[:, 2]).long().to(self.device)
        hh_embs = self.ent_h_embs(heads).to(self.device)
        ht_embs = self.ent_h_embs(tails).to(self.device)
        th_embs = self.ent_t_embs(heads).to(self.device)
        tt_embs = self.ent_t_embs(tails).to(self.device)
        r_embs = self.rel_embs(rels).to(self.device)
        r_inv_embs = self.rel_inv_embs(rels).to(self.device)

        scores1 = torch.sum(hh_embs * r_embs * tt_embs, dim=1)
        scores2 = torch.sum(ht_embs * r_inv_embs * th_embs, dim=1)
        return torch.clamp((scores1 + scores2) / 2, -20, 20)

    def forward_bce(self, batch):
        e1 = torch.tensor(batch[:, 0]).long().to(self.device)
        rel = torch.tensor(batch[:, 1]).long().to(self.device)
        hh_embs = self.ent_h_embs(e1).to(self.device)
        th_embs = self.ent_t_embs(e1).to(self.device)
        r_embs = self.rel_embs(rel).to(self.device)
        r_inv_embs = self.rel_inv_embs(rel).to(self.device)

        hh_embs = self.inp_drop(hh_embs)
        r_embs = self.inp_drop(r_embs)
        th_embs = self.inp_drop(th_embs)
        r_inv_embs = self.inp_drop(r_inv_embs)

        scores1 = hh_embs * r_embs
        scores2 = r_inv_embs * th_embs

        scores1 = torch.mm(self.ent_h_embs.weight, scores1.transpose(1, 0))
        scores2 = torch.mm(scores2, self.ent_t_embs.weight.transpose(1, 0))

        pred = torch.clamp((scores1.transpose(1, 0) + scores2) / 2, -20, 20)
        return torch.sigmoid(pred)

    def get_score(self, h, r, t):
        hh_embs = self.ent_h_embs(h).to(self.device)
        th_embs = self.ent_t_embs(h).to(self.device)
        r_embs = self.rel_embs(r).to(self.device)
        r_inv_embs = self.rel_inv_embs(r).to(self.device)

        hh_embs = self.inp_drop(hh_embs)
        r_embs = self.inp_drop(r_embs)
        th_embs = self.inp_drop(th_embs)
        r_inv_embs = self.inp_drop(r_inv_embs)

        scores1 = (hh_embs * r_embs).unsqueeze(0)
        scores2 = (r_inv_embs * th_embs).unsqueeze(0)

        scores1 = torch.mm(self.ent_h_embs.weight, scores1.transpose(1, 0))
        scores2 = torch.mm(scores2, self.ent_t_embs.weight.transpose(1, 0))

        return torch.clamp((scores1.transpose(1, 0) + scores2) / 2, -20, 20)

    def _regularization(self, heads, rels, tails):
        hh_embs = self.ent_h_embs(heads).to(self.device)
        ht_embs = self.ent_h_embs(tails).to(self.device)
        th_embs = self.ent_t_embs(heads).to(self.device)
        tt_embs = self.ent_t_embs(tails).to(self.device)
        r_embs = self.rel_embs(rels).to(self.device)
        r_inv_embs = self.rel_inv_embs(rels).to(self.device)
        regul = (torch.mean(hh_embs ** 2) + torch.mean(ht_embs ** 2) + torch.mean(th_embs ** 2) +
                 torch.mean(tt_embs ** 2) + torch.mean(r_embs ** 2) + torch.mean(r_inv_embs ** 2)) / 6
        return regul

    def _logistic_pairwise(self, y_pos, y_neg, neg_ratio, gamma=1.0):
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0] / P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        exp = torch.exp(y_neg - y_pos)
        loss = torch.mean(gamma * torch.log(1 + exp))
        return loss

    def _softplus(self, scores, neg_ratio, beta, label):
        criterion = nn.Softplus(beta=beta)
        loss = torch.mean(criterion(-scores * label))
        return loss

    def _loss(self, y_pos, y_neg, neg_ratio, gamma):
        criterion = nn.MarginRankingLoss(margin=gamma)
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0] / P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        target = Variable(torch.from_numpy(
            np.ones(P * (int(y_neg.shape[0] / P)), dtype=np.int32))).to(self.device)
        # '-1' for target means y_neg value should be higher than the one of y_pos.
        loss = criterion(y_pos, y_neg, target)
        return loss

    def _bce_vanilla(self, pos_batch):
        preds = self.forward_bce(pos_batch).to(self.device)
        ts = torch.tensor(pos_batch[:, 2]).long().to(self.device)
        labels = self.get_batch(pos_batch.shape[0], ts).to(self.device)
        return self.BCE(preds, labels)

    def _bce_sem(self, pos_batch, labels):
        preds = self.forward_bce(pos_batch).to(self.device)
        return self.BCE(preds, labels)

    def get_batch(self, batch_size, batch_t):
        targets = torch.zeros(batch_size, self.num_ent).scatter_(
            1, batch_t.cpu().view(-1, 1).type(torch.int64), 1).to(self.device)
        return targets


class ComplEx(nn.Module):
    def __init__(
            self,
            num_ent,
            num_rel,
            emb_dim,
            device,
            args,
            margin=2.0,
            lmbda=0.0):
        super(ComplEx, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        if args.lossfunc == 'bce':
            self.num_rel = num_rel * 2
        self.emb_dim = emb_dim
        self.gamma = margin
        self.lmbda = lmbda
        self.device = device
        self.BCE = nn.BCELoss()
        torch.manual_seed(7)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.ent_re_embeddings = nn.Embedding(
            self.num_ent, self.emb_dim).to(
            self.device)
        self.ent_im_embeddings = nn.Embedding(
            self.num_ent, self.emb_dim).to(
            self.device)
        self.rel_re_embeddings = nn.Embedding(
            self.num_rel, self.emb_dim).to(
            self.device)
        self.rel_im_embeddings = nn.Embedding(
            self.num_rel, self.emb_dim).to(
            self.device)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    def _calc(self, e_re_h, e_im_h, r_re, r_im, e_re_t, e_im_t):
        return torch.sum(
            r_re *
            e_re_h *
            e_re_t +
            r_re *
            e_im_h *
            e_im_t +
            r_im *
            e_re_h *
            e_im_t -
            r_im *
            e_im_h *
            e_re_t,
            1,
            False)

    def _loss(self, y_pos, y_neg, neg_ratio, gamma):
        criterion = nn.MarginRankingLoss(margin=gamma)
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0] / P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        target = Variable(torch.from_numpy(
            np.ones(P * (int(y_neg.shape[0] / P)), dtype=np.int32))).to(self.device)
        loss = criterion(y_pos, y_neg, target)
        return loss

    def _softplus(self, scores, neg_ratio, beta, label):
        criterion = nn.Softplus(beta=beta)
        loss = torch.mean(criterion(-scores * label))
        return loss

    def _logistic_pairwise(self, y_pos, y_neg, neg_ratio, gamma=1.0):
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0] / P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        exp = torch.exp(y_neg - y_pos)
        loss = torch.mean(gamma * torch.log(1 + exp))
        return loss

    def forward2(self, hs, rs, ts):
        e_re_h, e_im_h = self.ent_re_embeddings(hs).to(
            self.device), self.ent_im_embeddings(hs).to(
            self.device)
        e_re_t, e_im_t = self.ent_re_embeddings(ts).to(
            self.device), self.ent_im_embeddings(ts).to(
            self.device)
        r_re, r_im = self.rel_re_embeddings(rs).to(
            self.device), self.rel_im_embeddings(rs).to(
            self.device)
        score = self._calc(e_re_h, e_im_h, r_re, r_im, e_re_t, e_im_t)
        return score

    def forward(self, batch):
        hs = torch.tensor(batch[:, 0]).long().to(self.device)
        rs = torch.tensor(batch[:, 1]).long().to(self.device)
        ts = torch.tensor(batch[:, 2]).long().to(self.device)
        e_re_h, e_im_h = self.ent_re_embeddings(hs).to(
            self.device), self.ent_im_embeddings(hs).to(
            self.device)
        e_re_t, e_im_t = self.ent_re_embeddings(ts).to(
            self.device), self.ent_im_embeddings(ts).to(
            self.device)
        r_re, r_im = self.rel_re_embeddings(rs).to(
            self.device), self.rel_im_embeddings(rs).to(
            self.device)
        score = self._calc(
            e_re_h,
            e_im_h,
            r_re,
            r_im,
            e_re_t,
            e_im_t).to(
            self.device)
        return score

    def forward_bce(self, batch):
        e1 = torch.tensor(batch[:, 0]).long().to(self.device)
        rel = torch.tensor(batch[:, 1]).long().to(self.device)
        e1_embedded_real = self.ent_re_embeddings(e1).squeeze()
        rel_embedded_real = self.rel_re_embeddings(rel).squeeze()
        e1_embedded_img = self.ent_im_embeddings(e1).squeeze()
        rel_embedded_img = self.rel_im_embeddings(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        realrealreal = torch.mm(
            e1_embedded_real *
            rel_embedded_real,
            self.ent_re_embeddings.weight.transpose(
                1,
                0))
        realimgimg = torch.mm(
            e1_embedded_real *
            rel_embedded_img,
            self.ent_im_embeddings.weight.transpose(
                1,
                0))
        imgrealimg = torch.mm(
            e1_embedded_img *
            rel_embedded_real,
            self.ent_im_embeddings.weight.transpose(
                1,
                0))
        imgimgreal = torch.mm(
            e1_embedded_img *
            rel_embedded_img,
            self.ent_re_embeddings.weight.transpose(
                1,
                0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred

    def get_score(self, h, r, t):
        e1_embedded_real = self.ent_re_embeddings(h).squeeze()
        rel_embedded_real = self.rel_re_embeddings(r).squeeze()
        e1_embedded_img = self.ent_im_embeddings(h).squeeze()
        rel_embedded_img = self.rel_im_embeddings(r).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        realrealreal = torch.mm(
            (e1_embedded_real * rel_embedded_real).unsqueeze(0),
            self.ent_re_embeddings.weight.transpose(
                1,
                0))
        realimgimg = torch.mm(
            (e1_embedded_real * rel_embedded_img).unsqueeze(0),
            self.ent_im_embeddings.weight.transpose(
                1,
                0))
        imgrealimg = torch.mm(
            (e1_embedded_img * rel_embedded_real).unsqueeze(0),
            self.ent_im_embeddings.weight.transpose(
                1,
                0))
        imgimgreal = torch.mm(
            (e1_embedded_img * rel_embedded_img).unsqueeze(0),
            self.ent_re_embeddings.weight.transpose(
                1,
                0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal

        return pred

    def _regularization(self, hs, rs, ts):
        e_re_h, e_im_h = self.ent_re_embeddings(hs).to(
            self.device), self.ent_im_embeddings(hs).to(
            self.device)
        e_re_t, e_im_t = self.ent_re_embeddings(ts).to(
            self.device), self.ent_im_embeddings(ts).to(
            self.device)
        r_re, r_im = self.rel_re_embeddings(rs).to(
            self.device), self.rel_im_embeddings(rs).to(
            self.device)
        regul = (torch.mean(e_re_h ** 2) +
                 torch.mean(e_im_h ** 2) +
                 torch.mean(e_re_t ** 2) +
                 torch.mean(e_im_t ** 2) +
                 torch.mean(r_re ** 2) +
                 torch.mean(r_im ** 2)) / 6
        return regul

    def _bce_vanilla(self, pos_batch):
        preds = self.forward_bce(pos_batch).to(self.device)
        ts = torch.tensor(pos_batch[:, 2]).long().to(self.device)
        labels = self.get_batch(pos_batch.shape[0], ts).to(self.device)
        return self.BCE(preds, labels)

    def _bce_sem(self, pos_batch, labels):
        preds = self.forward_bce(pos_batch).to(self.device)
        return self.BCE(preds, labels)

    def get_batch(self, batch_size, batch_t):
        targets = torch.zeros(batch_size, self.num_ent).scatter_(
            1, batch_t.cpu().view(-1, 1).type(torch.int64), 1).to(self.device)
        return targets


class TuckER(nn.Module):
    def __init__(self, num_ent, num_rel, args, device):
        super(TuckER, self).__init__()
        self.args = args
        self.num_ent = num_ent
        self.num_rel = num_rel * 2
        self.dim_e = args.dim_e
        self.dim_r = args.dim_r
        self.device = device
        torch.manual_seed(7)
        self.ent_embeddings = torch.nn.Embedding(
            self.num_ent, self.dim_e).to(self.device)
        self.rel_embeddings = torch.nn.Embedding(
            self.num_rel, self.dim_r).to(self.device)
        xavier_normal_(self.ent_embeddings.weight.data)
        xavier_normal_(self.rel_embeddings.weight.data)
        self.W = torch.nn.Parameter(
            torch.tensor(
                np.random.uniform(
                    -1,
                    1,
                    (self.dim_r,
                     self.dim_e,
                     self.dim_e)),
                dtype=torch.float,
                requires_grad=True).to(
                self.device))

        self.input_dropout = torch.nn.Dropout(
            self.args.input_dropout).to(
            self.device)
        self.hidden_dropout1 = torch.nn.Dropout(
            self.args.hidden_dropout1).to(
            self.device)
        self.hidden_dropout2 = torch.nn.Dropout(
            self.args.hidden_dropout2).to(
            self.device)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(self.dim_e).to(self.device)
        self.bn1 = torch.nn.BatchNorm1d(self.dim_e).to(self.device)

    def calc(self, e1, r):
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.ent_embeddings.weight.transpose(1, 0))
        return x

    def forward(self, batch):
        hs = torch.tensor(batch[:, 0]).long().to(self.device)
        rs = torch.tensor(batch[:, 1]).long().to(self.device)

        e1 = self.ent_embeddings(hs).to(self.device)
        r = self.rel_embeddings(rs).to(self.device)
        pred = torch.sigmoid(self.calc(e1, r))
        return pred

    def _bce_vanilla(self, pos_batch):
        preds = self.forward(pos_batch).to(self.device)
        ts = torch.tensor(pos_batch[:, 2]).long().to(self.device)
        labels = self.get_batch(pos_batch.shape[0], ts).to(self.device)
        return self.loss(preds, labels)

    def _bce_sem(self, pos_batch, labels):
        preds = self.forward(pos_batch).to(self.device)
        return self.loss(preds, labels)

    def get_batch(self, batch_size, batch_t):
        targets = torch.zeros(batch_size, self.num_ent).scatter_(
            1, batch_t.cpu().view(-1, 1).type(torch.int64), 1).to(self.device)
        return targets

    def get_score(self, h, r, t):
        e1 = self.ent_embeddings(h).unsqueeze(0).to(self.device)
        r = self.rel_embeddings(r).unsqueeze(0).to(self.device)
        # je mets un "-" ici comme dans muKG. Si n'améliore pas, essayer de "-"
        # directement dans calc()
        pred = self.calc(e1, r)
        return pred

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embeddings(hs).to(
            self.device), self.rel_embeddings(rs).to(
            self.device), self.ent_embeddings(ts).to(
            self.device)
        regul = (torch.mean(e_hs ** 2) + torch.mean(e_ts ** 2) +
                 torch.mean(e_rs ** 2)) / 3
        return regul


### CONVOLUTIONAL MODELS ###
class ConvE(nn.Module):
    def __init__(self, num_ent, num_rel, args, device):
        super(ConvE, self).__init__()
        self.args = args
        self.device = device
        self.num_ent = num_ent
        self.num_rel = num_rel * 2
        torch.manual_seed(7)
        self.ent_embeddings = nn.Embedding(
            self.num_ent,
            self.args.dim,
            padding_idx=0).to(
            self.device)
        self.rel_embeddings = nn.Embedding(
            self.num_rel,
            self.args.dim,
            padding_idx=0).to(
            self.device)
        self.inp_drop = nn.Dropout(self.args.input_drop).to(self.device)
        self.hidden_drop = nn.Dropout(self.args.hidden_drop).to(self.device)
        self.feature_map_drop = nn.Dropout2d(
            self.args.feat_drop).to(
            self.device)
        self.loss = nn.BCELoss()
        self.emb_dim1 = self.args.embedding_shape1
        self.emb_dim2 = self.args.dim // self.emb_dim1
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        self.conv1 = nn.Conv2d(
            1, 32, (3, 3), (1, 1), 0, bias=True).to(
            self.device)
        self.bn0 = nn.BatchNorm2d(1).to(self.device)
        self.bn1 = nn.BatchNorm2d(32).to(self.device)
        self.bn2 = nn.BatchNorm1d(self.args.dim).to(self.device)
        self.register_parameter(
            'b', nn.Parameter(
                torch.zeros(
                    self.num_ent).to(
                    self.device)))
        self.fc = nn.Linear(
            self.args.hidden_size,
            self.args.dim).to(
            self.device)

    def calc(self, e1_embedded, rel_embedded):
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.ent_embeddings.weight.transpose(1, 0))
        x += self.b.expand_as(x).to(self.device)
        return x

    def forward(self, batch):
        hs = torch.tensor(batch[:, 0]).long().to(self.device)
        rs = torch.tensor(batch[:, 1]).long().to(self.device)
        e1_embedded = self.ent_embeddings(
            hs).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        rel_embedded = self.rel_embeddings(
            rs).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        pred1 = self.calc(e1_embedded, rel_embedded)
        return torch.sigmoid(pred1)

    def calc_loss(self, hs, rs, ts):
        targets = self.get_batch(hs.shape[0], ts)
        e1_embedded = self.ent_embeddings(
            hs).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        rel_embedded = self.rel_embeddings(
            rs).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        pred1 = torch.sigmoid(self.calc(e1_embedded, rel_embedded))
        return self.loss(pred1, targets)

    def _bce_vanilla(self, pos_batch):
        preds = self.forward(pos_batch).to(self.device)
        ts = torch.tensor(pos_batch[:, 2]).long().to(self.device)
        labels = self.get_batch(pos_batch.shape[0], ts).to(self.device)
        return self.loss(preds, labels)

    def _bce_sem(self, pos_batch, labels):
        preds = self.forward(pos_batch).to(self.device)
        return self.loss(preds, labels)

    def get_batch(self, batch_size, batch_t):
        targets = torch.zeros(batch_size, self.num_ent).scatter_(
            1, batch_t.cpu().view(-1, 1).type(torch.int64), 1).to(self.device)
        return targets

    def get_score(self, h, r, t):
        e1_embedded = self.ent_embeddings(
            h).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        rel_embedded = self.rel_embeddings(
            r).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        pred = self.calc(e1_embedded, rel_embedded)
        return pred

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embeddings(hs).to(
            self.device), self.rel_embeddings(rs).to(
            self.device), self.ent_embeddings(ts).to(
            self.device)
        regul = (torch.mean(e_hs ** 2) + torch.mean(e_ts ** 2) +
                 torch.mean(e_rs ** 2)) / 3
        return regul
