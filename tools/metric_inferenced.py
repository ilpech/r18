import os
import numpy as np
import mxnet as mx
import shutil
from data_loader import DataLoader
import argparse
from tools import str2bool
import yaml
import json
import time
from cifar_wide_resnet import WideResNet, WideResNetBlock
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from tools import (
    debug,
    str2bool, 
    readSearchXlsxReport, 
    roundUp,
    norm_shifted_log,
    denorm_shifted_log,
    list2nonEmptyIds,
    listfind,
    is_number,
    ensure_folder,
    shuffle,
    setindxs
)
from gene_mapping import (
    mapping2dict, 
    uniprot_mapping_header, 
    uniq_nonempty_uniprot_mapping_header
)
import csv 

# isdebug = True
isdebug = False
# metric_file = '../trained/2export/resnet_regressor.007/r18_tissue29_preds_labels.txt'
metric_file = '../trained/2export/resnet_regressor.007/r18_tissue29_preds_labels_extended.txt'
uids = []
rna_exps = []
rna_values = []
prot_exps = []
labels = []
preds = []
with open(metric_file, 'r') as f:
    ds = f.readlines()
for d in ds:
    d = d[:-1]
    data = d.split('\t')
    uids.append(data[0])
    rna_exps.append(data[1])
    rna_values.append(float(data[2]))
    prot_exps.append(data[3])
    labels.append(float(data[4]))
    preds.append(float(data[5]))
prot_exps_alph = sorted(set(prot_exps))
print('{} experiments read'.format(len(labels)))
labels = mx.nd.array(labels)
preds = mx.nd.array(preds)
metric = mx.metric.PearsonCorrelation()
print('=============')
print('experiments metrics(PearsonCorrelation)::')
exps_metrics = []
for exp in prot_exps_alph:
    exp_metric = mx.metric.PearsonCorrelation()
    ids2use = [i for i in range(len(labels)) if prot_exps[i] == exp]
    exp_labels = mx.nd.array([labels[i].asscalar() for i in range(len(labels)) if i in ids2use])
    exp_preds = mx.nd.array([preds[i].asscalar() for i in range(len(preds)) if i in ids2use])
    exp_metric.update(exp_labels, exp_preds)
    _, v = exp_metric.get()
    print('. ({}){} = {}'.format(len(exp_labels), exp, v))
    exps_metrics.append(v)
    exp_metric.reset()
print('=============')
metric.update(labels, preds)
_, overall_metric = metric.get()
print('min metric(PearsonCorrelation) {}'.format(max(exps_metrics)))
print('max metric(PearsonCorrelation) {}'.format(min(exps_metrics)))
print('=============')
print('overall metric(PearsonCorrelation) {}'.format(overall_metric))
print('=============')