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
        if self.model_name == 'TransN':
            self.model_sem = TransN(
                dataset.num_ent(),
                dataset.num_rel(),
                dataset.num_class(),
                dataset.ents2classes_matrix,
                args.dim,
                self.device)
        if self.model_name == 'TransE':
            self.model_sem = TransE(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device)
        if self.model_name == 'TransH':
            self.model_sem = TransH(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device)
        if self.model_name == 'TransD':
            self.model_sem = TransD(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device)
        if self.model_name == 'DistMult':
            self.model_sem = DistMult(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device,
                args)
        if self.model_name == 'ComplEx':
            self.model_sem = ComplEx(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device,
                args)
        if self.model_name == 'SimplE':
            self.model_sem = SimplE(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device,
                self.lmbda,
                args)
        if self.model_name == 'TuckER':
            self.model_sem = TuckER(
                dataset.num_ent(),
                dataset.num_rel(),
                args,
                self.device)
        if self.model_name == 'ConvE':
            self.model_sem = ConvE(
                dataset.num_ent(),
                dataset.num_rel(),
                args,
                self.device)
        self.dataset = dataset
        self.args = args
        self.directory = "models/" + self.dataset.name + "/" + self.model_name + "/"
        self.lossfunc = args.lossfunc
        self.labelsem = args.labelsem

    def train(self):
        self.model_sem.train()
        optimizer_sem = torch.optim.Adam(
            self.model_sem.parameters(),
            lr=self.args.lr
        )
        if not self.args.resume_training:
            start = 1
        else:
            start = self.args.resume_epoch + 1
        for epoch in range(start, self.args.ne + 1):
            last_batch = False
            total_loss_sem = 0.0
            while not last_batch:
                batch_pos = self.dataset.next_pos_batch(self.args.batch_size)
                if self.model_name != 'ConvE' and self.model_name != 'TuckER':
                    neg_batch_sem, neg_batch_nonsem = self.dataset.neg_picking(
                        batch_pos, neg_ratio=1)
                last_batch = self.dataset.was_last_batch()
                optimizer_sem.zero_grad()
                if self.model_name != 'ConvE' and self.model_name != 'TuckER':
                    pos_scores_s, neg_scores_sem_s, neg_scores_nonsem_s = self.model_sem.forward(
                        batch_pos), self.model_sem.forward(neg_batch_sem), self.model_sem.forward(neg_batch_nonsem)
                if self.lossfunc == 'bce':
                    if self.args.bce_alpha == 1:
                        x = random.random()
                        if x < self.args.alpha:
                            sem_labels = self.dataset.labelling(
                                batch_pos, lmbda=1)
                            loss_sem = self.model_sem._bce_sem(
                                batch_pos, sem_labels)
                        else:
                            loss_sem = self.model_sem._bce_vanilla(batch_pos)

                    else:
                        sem_labels = self.dataset.labelling(
                            batch_pos, lmbda=self.labelsem)
                        loss_sem = self.model_sem._bce_sem(
                            batch_pos, sem_labels)

                elif self.lossfunc == 'softplus':
                    loss_pos_s = self.model_sem._softplus(
                        pos_scores_s, self.args.neg_ratio, beta=1, label=1)
                    loss_nonsem_s = self.model_sem._softplus(
                        neg_scores_nonsem_s, self.args.neg_ratio, beta=1, label=-1)
                    if self.args.softplus_epsilon == 1:
                        loss_sem_s = self.model_sem._softplus(
                            neg_scores_sem_s, self.args.neg_ratio, beta=1, label=self.args.labelsem)
                    else:
                        x = random.random()
                        if x < self.args.alpha:
                            loss_sem_s = self.model_sem._softplus(
                                neg_scores_sem_s, self.args.neg_ratio, beta=1, label=1)
                        else:
                            loss_sem_s = self.model_sem._softplus(
                                neg_scores_sem_s, self.args.neg_ratio, beta=1, label=-1)
                    loss_sem = loss_pos_s + loss_sem_s + loss_nonsem_s

                elif self.lossfunc == 'pairwise':
                    loss_sem_s = self.model_sem._loss(
                        pos_scores_s, neg_scores_sem_s, self.args.neg_ratio, gamma=self.args.gamma2)
                    loss_nonsem_s = self.model_sem._loss(
                        pos_scores_s, neg_scores_nonsem_s, self.args.neg_ratio, gamma=self.args.gamma1)
                    loss_sem = loss_sem_s + loss_nonsem_s

                elif self.lossfunc == 'logistic_pairwise':
                    loss_sem_s = self.model_sem._logistic_pairwise(
                        pos_scores_s, neg_scores_sem_s, self.args.neg_ratio, gamma=0.5)
                    loss_nonsem_s = self.model_sem._logistic_pairwise(
                        pos_scores_s, neg_scores_nonsem_s, self.args.neg_ratio, gamma=1.0)
                    loss_sem = loss_sem_s + loss_nonsem_s

                if self.args.reg != 0.0:
                    batch = np.concatenate(
                        (batch_pos, neg_batch_sem, neg_batch_nonsem), axis=0)
                    batch = torch.tensor(batch)
                    loss_sem += self.args.reg * self.model_sem._regularization(batch[:, 0].to(
                        self.device), batch[:, 1].to(self.device), batch[:, 2].to(self.device))

                loss_sem.backward()
                optimizer_sem.step()
                total_loss_sem += loss_sem.cpu().item()

            if epoch % self.args.save_each == 0:
                print(
                    "Sem Loss in iteration " +
                    str(epoch) +
                    ": " +
                    str(total_loss_sem))

            if epoch % self.args.save_each == 0:
                self.save_model(self.model_name, epoch)
                if self.args.monitor_metrics == 1:
                    if self.model_name == 'ConvE':
                        model_path_sem = self.directory + "sem__dim=" + str(self.args.dim) + \
                            "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_input_drop=" + str(self.args.input_drop) + "_hid_dropout=" + str(self.args.hidden_drop) + \
                            "_feat_drop=" + str(self.args.feat_drop) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                            "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(epoch) + ".pt"
                    elif self.model_name == 'TuckER':
                        model_path_sem = self.directory + "sem__dim_e=" + str(self.args.dim_e) + "_dim_r=" + str(self.args.dim_r) + \
                            "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_input_drop=" + str(self.args.input_dropout) + "_hid_dropout1=" + str(self.args.hidden_dropout1) + \
                            "_hid_dropout2=" + str(self.args.hidden_dropout2) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                            "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(epoch) + ".pt"
                    else:
                        model_path_sem = self.directory + "sem__dim=" + str(self.args.dim) + \
                            "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_reg=" + str(self.args.reg) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                            "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(epoch) + ".pt"

                    tester = Tester(
                        self.dataset, self.args, model_path_sem, "valid")
                    filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10, filt_mrr_h, filt_mrr_t, \
                        filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t, schema_CWA, schema_OWA, ext_CWA, schema_WUP = tester.calc_valid_mrr()

    def save_model(self, model, chkpnt):
        print("Saving the model")
        directory = "models/" + self.dataset.name + "/" + model + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.model_name == 'ConvE':
            torch.save(self.model_sem.state_dict(), directory +
                       "sem__dim=" +
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
            torch.save(self.model_sem.state_dict(), directory +
                       "sem__dim_e=" +
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
            torch.save(self.model_sem.state_dict(), directory +
                       "sem__dim=" +
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
                model_path = directory + "sem__dim=" + str(self.args.dim) + \
                    "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_input_drop=" + str(self.args.input_drop) + "_hid_dropout=" + str(self.args.hidden_drop) + \
                    "_feat_drop=" + str(self.args.feat_drop) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                    "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(resume_epoch) + ".pt"
            elif self.model_name == 'TuckER':
                model_path = directory + "sem__dim_e=" + str(self.args.dim_e) + "_dim_r=" + str(self.args.dim_r) + \
                    "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_input_drop=" + str(self.args.input_dropout) + "_hid_dropout1=" + str(self.args.hidden_dropout1) + \
                    "_hid_dropout2=" + str(self.args.hidden_dropout2) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                    "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(resume_epoch) + ".pt"
            else:
                model_path = directory + "sem__dim=" + str(self.args.dim) + \
                    "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_reg=" + str(self.args.reg) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                    "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(resume_epoch) + ".pt"
        else:
            if self.model_name == 'ConvE':
                model_path = directory + "sem__dim=" + str(self.args.dim) + \
                    "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_input_drop=" + str(self.args.input_drop) + "_hid_dropout=" + str(self.args.hidden_drop) + \
                    "_feat_drop=" + str(self.args.feat_drop) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                    "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(resume_epoch) + ".pt"
            elif self.model_name == 'TuckER':
                model_path = directory + "sem__dim_e=" + str(self.args.dim_e) + "_dim_r=" + str(self.args.dim_r) + \
                    "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_input_drop=" + str(self.args.input_dropout) + "_hid_dropout1=" + str(self.args.hidden_dropout1) + \
                    "_hid_dropout2=" + str(self.args.hidden_dropout2) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                    "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(resume_epoch) + ".pt"
            else:
                model_path = directory + "sem__dim=" + str(self.args.dim) + \
                    "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_reg=" + str(self.args.reg) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                    "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(resume_epoch) + ".pt"

        print('Resuming from ' + str(model_path))
        self.model_sem.load_state_dict(torch.load(model_path))
        self.train()
