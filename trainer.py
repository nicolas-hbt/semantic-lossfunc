from dataset import Dataset
from tester import Tester
from models import *
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import random
from sklearn.utils import shuffle
torch.manual_seed(7)
random.seed(7)


class Trainer:
    def __init__(self, dataset, model_name, args):
        self.device = args.device
        self.model_name = model_name
        self.lmbda = args.reg
        torch.manual_seed(7)
        random.seed(7)
        if self.model_name == 'TransN':
            self.model_vanilla = TransN(
                dataset.num_ent(),
                dataset.num_rel(),
                dataset.num_class(),
                dataset.ents2classes_matrix,
                args.dim,
                self.device)
            self.model_sem = copy.deepcopy(self.model_vanilla)
        if self.model_name == 'TransE':
            self.model_vanilla = TransE(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device)
            self.model_sem = copy.deepcopy(self.model_vanilla)
        if self.model_name == 'TransH':
            self.model_vanilla = TransH(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device)
            self.model_sem = copy.deepcopy(self.model_vanilla)
        if self.model_name == 'TransD':
            self.model_vanilla = TransD(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device)
            self.model_sem = copy.deepcopy(self.model_vanilla)
        if self.model_name == 'DistMult':
            self.model_vanilla = DistMult(
                dataset.num_ent(), dataset.num_rel(), args.dim, self.device)
            self.model_sem = copy.deepcopy(self.model_vanilla)
        if self.model_name == 'ComplEx':
            self.model_vanilla = ComplEx(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device)
            self.model_sem = copy.deepcopy(self.model_vanilla)
        if self.model_name == 'SimplE':
            self.model_vanilla = SimplE(
                dataset.num_ent(),
                dataset.num_rel(),
                args.dim,
                self.device,
                self.lmbda)
            self.model_sem = copy.deepcopy(self.model_vanilla)
        if self.model_name == 'TuckER':
            self.model_vanilla = TuckER(
                dataset.num_ent(), dataset.num_rel(), args, self.device)
            self.model_sem = copy.deepcopy(self.model_vanilla)
        if self.model_name == 'ConvE':
            self.model_vanilla = ConvE(
                dataset.num_ent(), dataset.num_rel(), args, self.device)
            self.model_sem = copy.deepcopy(self.model_vanilla)
        if self.model_name == 'ConvKB2D':
            self.model_vanilla = ConvKB2D(
                dataset.num_ent(), dataset.num_rel(), args, self.device)
            self.model_sem = copy.deepcopy(self.model_vanilla)
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

        self.model_sem.train()
        optimizer_sem = torch.optim.Adam(
            self.model_sem.parameters(),
            lr=self.args.lr
        )

        for epoch in range(1, self.args.ne + 1):
            last_batch = False
            total_loss_vanilla = 0.0
            total_loss_sem = 0.0
            while not last_batch:
                batch_pos = self.dataset.next_pos_batch(self.args.batch_size)
                if self.model_name != 'ConvE' and self.model_name != 'TuckER':
                    neg_batch_sem, neg_batch_nonsem = self.dataset.neg_picking(
                        batch_pos, neg_ratio=1)
                last_batch = self.dataset.was_last_batch()
                optimizer_vanilla.zero_grad()
                optimizer_sem.zero_grad()
                if self.model_name != 'ConvE' and self.model_name != 'TuckER':
                    if self.model_name == 'SimplE' or self.model_name == 'ConvKB2D':
                        (pos_scores_v, regul_pos_scores_v) = self.model_vanilla.forward(
                            batch_pos)
                        (neg_scores_sem_v, regul_neg_scores_sem_v) = self.model_vanilla.forward(
                            neg_batch_sem)
                        (neg_scores_nonsem_v, regul_neg_scores_nonsem_v) = self.model_vanilla.forward(
                            neg_batch_nonsem)

                        (pos_scores_s,
                         regul_pos_scores_s) = self.model_sem.forward(batch_pos)
                        (neg_scores_sem_s, regul_neg_scores_sem_s) = self.model_sem.forward(
                            neg_batch_sem)
                        (neg_scores_nonsem_s, regul_neg_scores_nonsem_s) = self.model_sem.forward(
                            neg_batch_nonsem)
                    else:
                        pos_scores_v, neg_scores_sem_v, neg_scores_nonsem_v = self.model_vanilla.forward(
                            batch_pos), self.model_vanilla.forward(neg_batch_sem), self.model_vanilla.forward(neg_batch_nonsem)
                        pos_scores_s, neg_scores_sem_s, neg_scores_nonsem_s = self.model_sem.forward(
                            batch_pos), self.model_sem.forward(neg_batch_sem), self.model_sem.forward(neg_batch_nonsem)

                if self.lossfunc == 'bce':
                    loss_vanilla = self.model_vanilla._bce_vanilla(batch_pos)
                    sem_labels = self.dataset.labelling(
                        batch_pos, lmbda=self.labelsem)
                    loss_sem = self.model_sem._bce_sem(batch_pos, sem_labels)

                elif self.lossfunc == 'softplus':
                    if self.model_name == 'SimplE' or self.model_name == 'ConvKB2D':
                        loss_pos_v = self.model_vanilla._softplus(
                            pos_scores_v, regul_pos_scores_v, self.args.neg_ratio, beta=1, label=1)
                        loss_nonsem_v = self.model_vanilla._softplus(
                            neg_scores_nonsem_v, regul_neg_scores_sem_v, self.args.neg_ratio, beta=1, label=-1)
                        loss_sem_v = self.model_vanilla._softplus(
                            neg_scores_sem_v, regul_neg_scores_nonsem_v, self.args.neg_ratio, beta=1, label=-1)

                        loss_pos_s = self.model_sem._softplus(
                            pos_scores_s, regul_pos_scores_s, self.args.neg_ratio, beta=1, label=1)
                        loss_nonsem_s = self.model_sem._softplus(
                            neg_scores_nonsem_s,
                            regul_neg_scores_nonsem_s,
                            self.args.neg_ratio,
                            beta=1,
                            label=-1)
                        x = random.random()
                        if x < self.args.alpha:
                            loss_sem_s = self.model_sem._softplus(
                                neg_scores_sem_s, regul_neg_scores_sem_s, self.args.neg_ratio, beta=1, label=1)
                        else:
                            loss_sem_s = self.model_sem._softplus(
                                neg_scores_sem_s, regul_neg_scores_sem_s, self.args.neg_ratio, beta=1, label=-1)

                    else:
                        loss_pos_v = self.model_vanilla._softplus(
                            pos_scores_v, self.args.neg_ratio, beta=1, label=1)
                        loss_nonsem_v = self.model_vanilla._softplus(
                            neg_scores_nonsem_v, self.args.neg_ratio, beta=1, label=-1)
                        loss_sem_v = self.model_vanilla._softplus(
                            neg_scores_sem_v, self.args.neg_ratio, beta=1, label=-1)

                        loss_pos_s = self.model_sem._softplus(
                            pos_scores_s, self.args.neg_ratio, beta=1, label=1)
                        loss_nonsem_s = self.model_sem._softplus(
                            neg_scores_nonsem_s, self.args.neg_ratio, beta=1, label=-1)
                        x = random.random()
                        if x < self.args.alpha:
                            loss_sem_s = self.model_sem._softplus(
                                neg_scores_sem_s, self.args.neg_ratio, beta=1, label=1)
                        else:
                            loss_sem_s = self.model_sem._softplus(
                                neg_scores_sem_s, self.args.neg_ratio, beta=1, label=-1)

                    loss_vanilla = loss_pos_v + loss_sem_v + loss_nonsem_v
                    loss_sem = loss_pos_s + loss_sem_s + loss_nonsem_s

                elif self.lossfunc == 'pairwise':
                    loss_sem_v = self.model_vanilla._loss(
                        pos_scores_v, neg_scores_sem_v, self.args.neg_ratio, gamma=self.args.gamma1)
                    loss_nonsem_v = self.model_vanilla._loss(
                        pos_scores_v, neg_scores_nonsem_v, self.args.neg_ratio, gamma=self.args.gamma1)
                    loss_sem_s = self.model_sem._loss(
                        pos_scores_s, neg_scores_sem_s, self.args.neg_ratio, gamma=self.args.gamma2)
                    loss_nonsem_s = self.model_sem._loss(
                        pos_scores_s, neg_scores_nonsem_s, self.args.neg_ratio, gamma=self.args.gamma1)

                    loss_vanilla = loss_sem_v + loss_nonsem_v
                    loss_sem = loss_sem_s + loss_nonsem_s

                elif self.lossfunc == 'logistic_pairwise':
                    loss_sem_v = self.model_vanilla._logistic_pairwise(
                        pos_scores_v, neg_scores_sem_v, self.args.neg_ratio, gamma=1.0)
                    loss_nonsem_v = self.model_vanilla._logistic_pairwise(
                        pos_scores_v, neg_scores_nonsem_v, self.args.neg_ratio, gamma=1.0)
                    loss_sem_s = self.model_sem._logistic_pairwise(
                        pos_scores_s, neg_scores_sem_s, self.args.neg_ratio, gamma=0.5)
                    loss_nonsem_s = self.model_sem._logistic_pairwise(
                        pos_scores_s, neg_scores_nonsem_s, self.args.neg_ratio, gamma=1.0)

                    loss_vanilla = loss_sem_v + loss_nonsem_v
                    loss_sem = loss_sem_s + loss_nonsem_s

                loss_vanilla.backward(retain_graph=True)
                optimizer_vanilla.step()
                total_loss_vanilla += loss_vanilla.cpu().item()

                loss_sem.backward(retain_graph=True)
                optimizer_sem.step()
                total_loss_sem += loss_sem.cpu().item()

            if epoch % self.args.save_each == 0:
                print(
                    "Vanilla Loss in iteration " +
                    str(epoch) +
                    ": " +
                    str(total_loss_vanilla) +
                    " | Constrained Loss: " +
                    str(total_loss_sem))

            if epoch % self.args.save_each == 0:
                self.save_model(self.model_name, epoch)
                if self.args.monitor_metrics == 1:
                    model_path_vanilla = self.directory + "vanilla__dim=" + str(self.args.dim) + \
                        "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_reg=" + str(self.args.reg) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                        "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(epoch) + ".pt"
                    model_path_sem = self.directory + "sem__dim=" + str(self.args.dim) + \
                        "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_reg=" + str(self.args.reg) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                        "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(epoch) + ".pt"
                    tester = Tester(
                        self.dataset,
                        self.args,
                        model_path_vanilla,
                        "valid")
                    filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10, filt_mrr_h, filt_mrr_t, \
                        filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t, schema_CWA, schema_OWA, ext_CWA, schema_WUP = tester.calc_valid_mrr()
                    tester = Tester(
                        self.dataset, self.args, model_path_sem, "valid")
                    filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10, filt_mrr_h, filt_mrr_t, \
                        filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t, schema_CWA, schema_OWA, ext_CWA, schema_WUP = tester.calc_valid_mrr()

    def save_model(self, model, chkpnt):
        print("Saving the model")
        directory = "models/" + self.dataset.name + "/" + model + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
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
            model_path = directory + "dim=" + str(self.args.dim) + \
                "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_reg=" + str(self.args.reg) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(resume_epoch) + ".pt"
        else:
            model_path = directory + "dim=" + str(self.args.dim) + \
                "_loss=" + str(self.args.lossfunc) + "_lr=" + str(self.args.lr) + "_reg=" + str(self.args.reg) + "_neg=" + str(self.args.neg_ratio) + "_bs=" + str(self.args.batch_size) + \
                "_gamma1=" + str(self.args.gamma1) + "_gamma2=" + str(self.args.gamma2) + "_labelsem=" + str(self.args.labelsem) + "_alpha=" + str(self.args.alpha) + "__epoch=" + str(resume_epoch) + ".pt"

        print('Resuming from ' + str(model_path))
        self.model.load_state_dict(torch.load(model_path))
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr
        )