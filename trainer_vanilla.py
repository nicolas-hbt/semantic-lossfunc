from dataset import Dataset
from tester import Tester
from models import *
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from sklearn.utils import shuffle
torch.manual_seed(7)
random.seed(7)

class Trainer:
    def __init__(self, dataset, model_name, args):
        torch.manual_seed(7)
        random.seed(7)
        self.device = args.device
        self.model_name = model_name
        self.lmbda = args.reg
        if self.model_name == 'TransE':
            self.model_vanilla = TransE(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device)
        if self.model_name == 'TransH':
            self.model_vanilla = TransH(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device)
        if self.model_name == 'TransD':
            self.model_vanilla = TransD(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device)
        if self.model_name == 'DistMult':
            self.model_vanilla = DistMult(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device,
                args)
        if self.model_name == 'ComplEx':
            self.model_vanilla = ComplEx(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device,
                args)
        if self.model_name == 'SimplE':
            self.model_vanilla = SimplE(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device,
                self.lmbda,
                args)
        if self.model_name == 'TuckER':
            self.model_vanilla = TuckER(
                dataset.num_ent(), dataset.num_rel(), args, self.device)
        if self.model_name == 'ConvE':
            self.model_vanilla = ConvE(
                dataset.num_ent(), dataset.num_rel(), args, self.device)
        self.dataset = dataset
        self.args = args
        self.directory = "models/" + self.dataset.name + "/" + self.model_name + "/"
        self.lossfunc = args.lossfunc
        self.labelsem = args.labelsem

    def train(self):
        self.model_vanilla.train()
        optimizer_vanilla = torch.optim.Adam(
            self.model_vanilla.parameters(),
            lr=self.args.lr
        )
        if not self.args.resume_training:
            start = 1
        else:
            start = self.args.resume_epoch + 1
        for epoch in range(start, self.args.ne + 1):
            last_batch = False
            total_loss_vanilla = 0.0
            while not last_batch:
                batch_pos = self.dataset.next_pos_batch(self.args.batch_size)
                if self.lossfunc != 'bce':
                    neg_batch_sem, neg_batch_nonsem = self.dataset.neg_picking(
                        batch_pos, neg_ratio=1)
                last_batch = self.dataset.was_last_batch()
                optimizer_vanilla.zero_grad()
                if self.lossfunc != 'bce':
                    pos_scores_v, neg_scores_sem_v, neg_scores_nonsem_v = self.model_vanilla.forward(
                        batch_pos), self.model_vanilla.forward(neg_batch_sem), self.model_vanilla.forward(neg_batch_nonsem)
                else:
                    loss_vanilla = self.model_vanilla._bce_vanilla(batch_pos)

                if self.lossfunc == 'softplus':
                    loss_pos_v = self.model_vanilla._softplus(
                        pos_scores_v, self.args.neg_ratio, beta=1, label=1)
                    loss_nonsem_v = self.model_vanilla._softplus(
                        neg_scores_nonsem_v, self.args.neg_ratio, beta=1, label=-1)
                    loss_sem_v = self.model_vanilla._softplus(
                        neg_scores_sem_v, self.args.neg_ratio, beta=1, label=-1)

                    loss_vanilla = loss_pos_v + loss_sem_v + loss_nonsem_v

                elif self.lossfunc == 'pairwise':
                    loss_sem_v = self.model_vanilla._loss(
                        pos_scores_v, neg_scores_sem_v, self.args.neg_ratio, gamma=self.args.gamma1)
                    loss_nonsem_v = self.model_vanilla._loss(
                        pos_scores_v, neg_scores_nonsem_v, self.args.neg_ratio, gamma=self.args.gamma1)

                    loss_vanilla = loss_sem_v + loss_nonsem_v

                elif self.lossfunc == 'logistic_pairwise':
                    loss_sem_v = self.model_vanilla._logistic_pairwise(
                        pos_scores_v, neg_scores_sem_v, self.args.neg_ratio, gamma=1.0)
                    loss_nonsem_v = self.model_vanilla._logistic_pairwise(
                        pos_scores_v, neg_scores_nonsem_v, self.args.neg_ratio, gamma=1.0)

                    loss_vanilla = loss_sem_v + loss_nonsem_v

                if self.args.reg != 0.0:
                    if self.lossfunc != 'bce':
                        batch = np.concatenate(
                            (batch_pos, neg_batch_sem, neg_batch_nonsem), axis=0)
                        batch = torch.tensor(batch)
                        loss_vanilla += self.args.reg * self.model_vanilla._regularization(batch[:, 0].to(
                            self.device), batch[:, 1].to(self.device), batch[:, 2].to(self.device))
                    else:
                        batch = torch.tensor(batch_pos)
                        loss_vanilla += self.args.reg * self.model_vanilla._regularization(batch[:, 0].to(
                            self.device), batch[:, 1].to(self.device), batch[:, 2].to(self.device))

                loss_vanilla.backward()
                optimizer_vanilla.step()
                total_loss_vanilla += loss_vanilla.cpu().item()

            if epoch % self.args.save_each == 0:
                print(
                    "Vanilla Loss in iteration " +
                    str(epoch) +
                    ": " +
                    str(total_loss_vanilla))

            if epoch % self.args.save_each == 0:
                self.save_model(self.model_name, epoch)
                if self.args.monitor_metrics == 1:
                    if self.model_name == 'ConvE':
                        model_path_vanilla = self.directory + "vanilla__dim=" + str(self.args.dim) + \
                            "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_input_drop=" + str(self.args.input_drop) + "_hid_dropout=" + str(self.args.hidden_drop) + \
                            "_feat_drop=" + str(self.args.feat_drop) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                            "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(epoch) + ".pt"
                    elif self.model_name == 'TuckER':
                        model_path_vanilla = self.directory + "vanilla__dim_e=" + str(self.args.dim_e) + "_dim_r=" + str(self.args.dim_r) + \
                            "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_input_drop=" + str(self.args.input_dropout) + "_hid_dropout1=" + str(self.args.hidden_dropout1) + \
                            "_hid_dropout2=" + str(self.args.hidden_dropout2) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                            "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(epoch) + ".pt"
                    else:
                        model_path_vanilla = self.directory + "vanilla__dim=" + str(self.args.dim) + \
                            "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_reg=" + str(self.args.reg) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                            "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(epoch) + ".pt"

                    tester = Tester(
                        self.dataset,
                        self.args,
                        model_path_vanilla,
                        "valid")
                    filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10, filt_mrr_h, filt_mrr_t, \
                        filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t, schema_CWA, schema_OWA, ext_CWA, schema_WUP = tester.calc_valid_mrr()

    def save_model(self, model, chkpnt):
        print("Saving the model")
        directory = "models/" + self.dataset.name + "/" + model + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.model_name == 'ConvE':
            torch.save(self.model_vanilla.state_dict(), directory +
                       "vanilla__dim=" +
                       str(self.args.dim) +
                       "_loss=" +
                       str(self.args.lossfunc) +
                       "_lr=" +
                       str(self.args.lr) +
                       "_input_drop=" +
                       str(self.args.input_drop) +
                       "_hid_dropout=" +
                       str(self.args.hidden_drop) +
                       "_feat_drop=" +
                       str(self.args.feat_drop) +
                       "_neg=" +
                       str(self.args.neg_ratio) +
                       "_bs=" +
                       str(self.args.batch_size) +
                       "_gamma1=" +
                       str(self.args.gamma1) +
                       "_gamma2=" +
                       str(self.args.gamma2) +
                       "_labelsem=" +
                       str(self.args.labelsem) +
                       "_alpha=" +
                       str(self.args.alpha) +
                       "__epoch=" +
                       str(chkpnt) +
                       ".pt")
        elif self.model_name == 'TuckER':
            torch.save(self.model_vanilla.state_dict(), directory +
                       "vanilla__dim_e=" +
                       str(self.args.dim_e) +
                       "_dim_r=" +
                       str(self.args.dim_r) +
                       "_loss=" +
                       str(self.args.lossfunc) +
                       "_lr=" +
                       str(self.args.lr) +
                       "_input_drop=" +
                       str(self.args.input_dropout) +
                       "_hid_dropout1=" +
                       str(self.args.hidden_dropout1) +
                       "_hid_dropout2=" +
                       str(self.args.hidden_dropout2) +
                       "_neg=" +
                       str(self.args.neg_ratio) +
                       "_bs=" +
                       str(self.args.batch_size) +
                       "_gamma1=" +
                       str(self.args.gamma1) +
                       "_gamma2=" +
                       str(self.args.gamma2) +
                       "_labelsem=" +
                       str(self.args.labelsem) +
                       "_alpha=" +
                       str(self.args.alpha) +
                       "__epoch=" +
                       str(chkpnt) +
                       ".pt")
        else:
            torch.save(self.model_vanilla.state_dict(), directory +
                       "vanilla__dim=" +
                       str(self.args.dim) +
                       "_loss=" +
                       str(self.args.lossfunc) +
                       "_lr=" +
                       str(self.args.lr) +
                       "_reg=" +
                       str(self.args.reg) +
                       "_neg=" +
                       str(self.args.neg_ratio) +
                       "_bs=" +
                       str(self.args.batch_size) +
                       "_gamma1=" +
                       str(self.args.gamma1) +
                       "_gamma2=" +
                       str(self.args.gamma2) +
                       "_labelsem=" +
                       str(self.args.labelsem) +
                       "_alpha=" +
                       str(self.args.alpha) +
                       "__epoch=" +
                       str(chkpnt) +
                       ".pt")

    def resume_training(self):
        directory = "models/" + self.dataset.name + "/" + self.model_name + "/"
        resume_epoch = self.args.resume_epoch
        if resume_epoch == 0:
            resume_epoch = max([int(f[-11:].split('=')[-1].split('.')[0]) for f in os.listdir(
                "models/" + self.dataset.name + "/" + self.model_name + "/")])
            if self.model_name == 'ConvE':
                model_path = directory + "vanilla__dim=" + str(self.args.dim) + \
                    "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_input_drop=" + str(self.args.input_drop) + "_hid_dropout=" + str(self.args.hidden_drop) + \
                    "_feat_drop=" + str(self.args.feat_drop) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                    "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(resume_epoch) + ".pt"
            elif self.model_name == 'TuckER':
                model_path = directory + "vanilla__dim_e=" + str(self.args.dim_e) + "_dim_r=" + str(self.args.dim_r) + \
                    "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_input_drop=" + str(self.args.input_dropout) + "_hid_dropout1=" + str(self.args.hidden_dropout1) + \
                    "_hid_dropout2=" + str(self.args.hidden_dropout2) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                    "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(resume_epoch) + ".pt"
            else:
                model_path = directory + "vanilla__dim=" + str(self.args.dim) + \
                    "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_reg=" + str(self.args.reg) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                    "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(resume_epoch) + ".pt"
        else:
            if self.model_name == 'ConvE':
                model_path = directory + "vanilla__dim=" + str(self.args.dim) + \
                    "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_input_drop=" + str(self.args.input_drop) + "_hid_dropout=" + str(self.args.hidden_drop) + \
                    "_feat_drop=" + str(self.args.feat_drop) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                    "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(resume_epoch) + ".pt"
            elif self.model_name == 'TuckER':
                model_path = directory + "vanilla__dim_e=" + str(self.args.dim_e) + "_dim_r=" + str(self.args.dim_r) + \
                    "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_input_drop=" + str(self.args.input_dropout) + "_hid_dropout1=" + str(self.args.hidden_dropout1) + \
                    "_hid_dropout2=" + str(self.args.hidden_dropout2) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                    "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(resume_epoch) + ".pt"
            else:
                model_path = directory + "vanilla__dim=" + str(self.args.dim) + \
                    "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_reg=" + str(self.args.reg) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                    "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(resume_epoch) + ".pt"

        print('Resuming from ' + str(model_path))
        self.model_vanilla.load_state_dict(torch.load(model_path))
        self.train()
