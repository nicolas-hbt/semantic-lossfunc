from trainer_sem import Trainer
from bucket_tester import Tester
from dataset import Dataset
import numpy as np
import pandas as pd
import argparse
import os
import torch
import json
from datetime import datetime

date_today = datetime.today().strftime('%d-%m-%Y')


def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-epoch',
        default=400,
        type=int,
        help="number of epochs")
    parser.add_argument('-lr', default=0.001, type=float, help="learning rate")
    parser.add_argument('-reg', default=0.0, type=float,
                        help="l2 regularization parameter")
    parser.add_argument('-margin', default=2.0, type=float, help="margin")
    parser.add_argument(
        '-neg_sampler',
        default="rns",
        type=str,
        help="negative sampling strategy")
    parser.add_argument(
        '-dataset',
        default="Yago14k",
        type=str,
        help="dataset")
    parser.add_argument(
        '-model',
        default="TransE",
        type=str,
        help="knowledge graph embedding model")
    parser.add_argument(
        '-dim',
        default=100,
        type=int,
        help="embedding dimension")
    parser.add_argument(
        '-neg_ratio',
        default=1,
        type=int,
        help="number of negative examples per positive example")
    parser.add_argument(
        '-batch_size',
        default=1024,
        type=int,
        help="batch size")
    parser.add_argument(
        '-save_each',
        default=20,
        type=int,
        help="validate every k epochs")
    parser.add_argument(
        '-criterion_validation',
        default="mrr",
        type=str,
        help="criterion for keeping best epoch")
    parser.add_argument(
        '-metrics',
        default="all",
        type=str,
        help="metrics to compute on test set (sem|ranks|all)")
    parser.add_argument(
        '-pipeline',
        default="both",
        type=str,
        help="(train|test|both)")
    parser.add_argument(
        '-device',
        default="cuda:0",
        type=str,
        help="(cpu|cuda:0)")
    parser.add_argument(
        '-setting',
        default="CWA",
        type=str,
        help="CWA|OWA|both")
    parser.add_argument(
        '-sem',
        default="both",
        type=str,
        help="schema|extensional|both")
    parser.add_argument('-hierarchy', default=False, type=bool)
    parser.add_argument('-resume_training', default=False, type=bool)
    parser.add_argument(
        '-resume_epoch',
        default=0,
        type=int,
        help='epoch at which resuming training (0 means: last epoch the model was saved)')
    parser.add_argument('-test_one_epoch', default=False, type=bool)
    parser.add_argument(
        '-loss_strategy',
        default='vanilla',
        type=str,
        help='vanilla|sem')
    parser.add_argument('-lossfunc', default='pairwise', type=str)
    parser.add_argument('-monitor_metrics', default=0, type=int)
    parser.add_argument('-gamma1', default=2.0, type=float)
    parser.add_argument('-gamma2', default=0.5, type=float)
    parser.add_argument('-fast_testing', default=0, type=int)
    parser.add_argument('-labelsem', default=0.0001, type=float)
    parser.add_argument('-alpha', default=0.0, type=float)
    parser.add_argument('-softplus_epsilon', default=1, type=int)
    parser.add_argument('-bce_alpha', default=1, type=int)
    parser.add_argument('-buckets', default=3, type=int)

    # ConvE
    parser.add_argument('-input_drop', default=0.2, type=float)
    parser.add_argument('-hidden_drop', default=0.3, type=float)
    parser.add_argument('-feat_drop', default=0.3, type=float)
    parser.add_argument('-hidden_size', default=9728, type=int)
    parser.add_argument('-embedding_shape1', default=20, type=int)

    # TuckER
    parser.add_argument('-dim_e', default=100, type=int)
    parser.add_argument('-dim_r', default=100, type=int)
    parser.add_argument('-input_dropout', default=0.3, type=float)
    parser.add_argument('-hidden_dropout1', default=0.4, type=float)
    parser.add_argument('-hidden_dropout2', default=0.5, type=float)
    parser.add_argument('-label_smoothing', default=0.0, type=float)

    args = parser.parse_args()
    np.random.seed(7)
    torch.manual_seed(7)
    return args


if __name__ == '__main__':
    args = get_parameter()
    dataset = Dataset(args.dataset, args)
    model = args.model
    directory = "models/" + dataset.name + "/" + model + "/"
    if args.pipeline == 'both' or args.pipeline == 'test':
        print("------- Testing on the best epoch -------")
        if model == 'ConvE':
            best_model_path = directory + args.loss_strategy + "__dim=" + str(args.dim) + "_loss=" + str(args.lossfunc) + \
                "_lr=" + str(args.lr) + "_input_drop=" + str(args.input_drop) + "_hid_dropout=" + str(args.hidden_drop) + \
                "_feat_drop=" + str(args.feat_drop) + "_neg=" + str(args.neg_ratio) + "_bs=" + str(args.batch_size) + "_gamma1=" + str(args.gamma1) + "_gamma2=" + str(args.gamma2) + \
                "_labelsem=" + str(args.labelsem) + "_alpha=" + str(args.alpha) + "__epoch=" + str(args.epoch) + ".pt"

        elif model == 'TuckER':
            best_model_path = directory + args.loss_strategy + "__dim_e=" + str(args.dim_e) + "_dim_r=" + str(args.dim_r) + \
                "_loss=" + str(args.lossfunc) + "_lr=" + str(args.lr) + "_input_drop=" + str(args.input_dropout) + "_hid_dropout1=" + str(args.hidden_dropout1) + \
                "_hid_dropout2=" + str(args.hidden_dropout2) + "_neg=" + str(args.neg_ratio) + "_bs=" + str(args.batch_size) + \
                "_gamma1=" + str(args.gamma1) + "_gamma2=" + str(args.gamma2) + "_labelsem=" + str(args.labelsem) + "_alpha=" + str(args.alpha) + "__epoch=" + str(args.epoch) + ".pt"

        else:
            best_model_path = directory + args.loss_strategy + "__dim=" + str(args.dim) + "_loss=" + str(args.lossfunc) + \
                "_lr=" + str(args.lr) + "_reg=" + str(args.reg) + "_neg=" + str(args.neg_ratio) + "_bs=" + str(args.batch_size) +\
                "_gamma1=" + str(args.gamma1) + "_gamma2=" + str(args.gamma2) + "_labelsem=" + str(args.labelsem) + "_alpha=" + str(args.alpha) + "__epoch=" + str(args.epoch) + ".pt"
        tester = Tester(dataset, args, best_model_path, "test")
        if args.metrics == 'sem' or args.metrics == 'all':
            filt_mrr_h, filt_mrr_t, filt_mrr, filt_hit_h, filt_hit_t, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_10, \
                schema_CWA_h, schema_CWA_t, schema_CWA = tester.test()

            if args.sem == 'both':
                if args.setting == 'both' or args.setting == 'CWA':
                    best_ep_results = {}
                    best_ep_results['Epoch'] = args.epoch
                    best_ep_results['MRR'], best_ep_results['H@1'], best_ep_results['H@3'], best_ep_results['H@10'] = {}, {}, {}, {}
                    best_ep_results['S@1'], best_ep_results['S@3'], best_ep_results['S@10'] = {}, {}, {}
                    for b in range(1, args.buckets + 1):
                        best_ep_results["MRR"][b] = round(filt_mrr[b], 3)
                        best_ep_results["H@1"][b] = round(
                            filtered_hits_at_1[b], 3)
                        best_ep_results["H@3"][b] = round(
                            filtered_hits_at_3[b], 3)
                        best_ep_results["H@10"][b] = round(
                            filtered_hits_at_10[b], 3)
                        best_ep_results["S@1"][b] = round(
                            schema_CWA[b]['sem1'], 3)
                        best_ep_results["S@3"][b] = round(
                            schema_CWA[b]['sem3'], 3)
                        best_ep_results["S@10"][b] = round(
                            schema_CWA[b]['sem10'], 3)
            elif args.sem == 'schema':
                if args.setting == 'both' or args.setting == 'CWA':
                    best_ep_results = {}
                    best_ep_results['Epoch'] = args.epoch
                    best_ep_results['MRR'], best_ep_results['H@1'], best_ep_results['H@3'], best_ep_results['H@10'] = {}, {}, {}, {}
                    best_ep_results['S@1'], best_ep_results['S@3'], best_ep_results['S@10'] = {}, {}, {}
                    for b in range(1, args.buckets + 1):
                        best_ep_results["MRR"][b] = round(filt_mrr[b], 3)
                        best_ep_results["H@1"][b] = round(
                            filtered_hits_at_1[b], 3)
                        best_ep_results["H@3"][b] = round(
                            filtered_hits_at_3[b], 3)
                        best_ep_results["H@10"][b] = round(
                            filtered_hits_at_10[b], 3)
                        best_ep_results["S@1"][b] = round(
                            schema_CWA[b]['sem1'], 3)
                        best_ep_results["S@3"][b] = round(
                            schema_CWA[b]['sem3'], 3)
                        best_ep_results["S@10"][b] = round(
                            schema_CWA[b]['sem10'], 3)

            df_test = pd.DataFrame.from_dict([best_ep_results])
            print(df_test)
            path = 'bucket-results/' + dataset.name + "/" + args.model + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            if args.model == 'ConvE':
                df_test.to_csv(path +
                               '[' +
                               date_today +
                               ']best-epoch_results-ALL-METRICS-' +
                               args.dataset +
                               '-' +
                               model +
                               '-loss_str=' +
                               args.loss_strategy +
                               '-dim=' +
                               str(args.dim) +
                               "-loss=" +
                               str(args.lossfunc) +
                               '-lr=' +
                               str(args.lr) +
                               "-input_drop=" +
                               str(args.input_drop) +
                               "-hid_dropout=" +
                               str(args.hidden_drop) +
                               "-feat_drop=" +
                               str(args.feat_drop) +
                               '-neg=' +
                               str(args.neg_ratio) +
                               '-bs=' +
                               str(args.batch_size) +
                               "-gamma1=" +
                               str(args.gamma1) +
                               "-gamma2=" +
                               str(args.gamma2) +
                               "-labelsem=" +
                               str(args.labelsem) +
                               "-alpha=" +
                               str(args.alpha) +
                               '.csv')
                with open(path + '[' + date_today + ']best-epoch_results-ALL-METRICS-' + args.dataset + '-' + model + '-loss_str=' + args.loss_strategy + '-dim=' +
                          str(args.dim) + "-loss=" + str(args.lossfunc) + '-lr=' + str(args.lr) + "-input_drop=" + str(args.input_drop) + "-hid_dropout=" + str(args.hidden_drop) +
                          "-feat_drop=" + str(args.feat_drop) + '-neg=' + str(args.neg_ratio) + '-bs=' + str(args.batch_size) +
                          "-gamma1=" + str(args.gamma1) + "-gamma2=" + str(args.gamma2) + "-labelsem=" + str(args.labelsem) + "-alpha=" + str(args.alpha) + '.json', 'w') as fp:
                    json.dump(best_ep_results, fp)

            elif args.model == 'TuckER':
                df_test.to_csv(path +
                               '[' +
                               date_today +
                               ']best-epoch_results-ALL-METRICS-' +
                               args.dataset +
                               '-' +
                               model +
                               '-loss_str=' +
                               args.loss_strategy +
                               "-dim_e=" +
                               str(args.dim_e) +
                               "-dim_r=" +
                               str(args.dim_r) +
                               "-loss=" +
                               str(args.lossfunc) +
                               '-lr=' +
                               str(args.lr) +
                               "-input_drop=" +
                               str(args.input_dropout) +
                               "-hid_dropout1=" +
                               str(args.hidden_dropout1) +
                               "-hid_dropout2=" +
                               str(args.hidden_dropout2) +
                               '-neg=' +
                               str(args.neg_ratio) +
                               '-bs=' +
                               str(args.batch_size) +
                               "-gamma1=" +
                               str(args.gamma1) +
                               "-gamma2=" +
                               str(args.gamma2) +
                               "-labelsem=" +
                               str(args.labelsem) +
                               "-alpha=" +
                               str(args.alpha) +
                               '.csv')
                with open(path + '[' + date_today + ']best-epoch_results-ALL-METRICS-' + args.dataset + '-' + model + '-loss_str=' + args.loss_strategy + '-dim=' +
                          str(args.dim) + "-loss=" + str(args.lossfunc) + '-lr=' + str(args.lr) + "-input_drop=" + str(args.input_dropout) + "-hid_dropout1=" + str(args.hidden_dropout1) +
                          "-hid_dropout2=" + str(args.hidden_dropout2) + '-neg=' + str(args.neg_ratio) + '-bs=' + str(args.batch_size) +
                          "-gamma1=" + str(args.gamma1) + "-gamma2=" + str(args.gamma2) + "-labelsem=" + str(args.labelsem) + "-alpha=" + str(args.alpha) + '.json', 'w') as fp:
                    json.dump(best_ep_results, fp)

            else:
                df_test.to_csv(path +
                               '[' +
                               date_today +
                               ']best-epoch_results-ALL-METRICS-' +
                               args.dataset +
                               '-' +
                               model +
                               '-loss_str=' +
                               args.loss_strategy +
                               '-dim=' +
                               str(args.dim) +
                               "-loss=" +
                               str(args.lossfunc) +
                               '-lr=' +
                               str(args.lr) +
                               "-reg=" +
                               str(args.reg) +
                               '-neg=' +
                               str(args.neg_ratio) +
                               '-bs=' +
                               str(args.batch_size) +
                               "-gamma1=" +
                               str(args.gamma1) +
                               "-gamma2=" +
                               str(args.gamma2) +
                               "-labelsem=" +
                               str(args.labelsem) +
                               "-alpha=" +
                               str(args.alpha) +
                               '.csv')
                with open(path + '[' + date_today + ']best-epoch_results-ALL-METRICS-' + args.dataset + '-' + model + '-loss_str=' + args.loss_strategy + '-dim=' +
                          str(args.dim) + "-loss=" + str(args.lossfunc) + '-lr=' + str(args.lr) + "-reg=" + str(args.reg) + '-neg=' + str(args.neg_ratio) + '-bs=' + str(args.batch_size) +
                          "-gamma1=" + str(args.gamma1) + "-gamma2=" + str(args.gamma2) + "-labelsem=" + str(args.labelsem) + "-alpha=" + str(args.alpha) + '.json', 'w') as fp:
                    json.dump(best_ep_results, fp)
