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

def load_exported(sym_path, params_path, ctx):
    """
    load model
    self.sym_path and self.params_path are using if given are not exist
    """
    if not os.path.isfile(params_path):
        print('Check params path', params_path)
        raise FileNotFoundError(params_path)
    if not os.path.isfile(sym_path):
        print('Check sym path', sym_path)
        raise FileNotFoundError(sym_path)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mx.gluon.nn.SymbolBlock.imports(
            sym_path, 
            ['data'], 
            params_path, 
            ctx=ctx
        )
data_loader = DataLoader('../config/train.yaml')
data_loader.loadTissue29data2genes(
    '../data/liver_hepg2/tissue29.1k_rna.tsv',
    '../data/liver_hepg2/tissue29.1k_prot.tsv',
    # '../data/liver_hepg2/tissue29.05k_rna.tsv',
    # '../data/liver_hepg2/tissue29.05k_prot.tsv',
    create_new_genes=True,
    isdebug=isdebug
)
try:
    ctx = mx.gpu()
    a = mx.nd.array([[0]], ctx=ctx)
    print('successfully created gpu array -> using gpu')
except:
    ctx = mx.cpu()
    print('cannot create gpu array -> using cpu')
net_name = 'resnet_regressor.007'
epoch = 95
params_path = '../trained/2export/{}'.format(net_name)
config_p='{}/{}_config.json'.format(params_path, net_name)
with open(config_p, 'r') as f:
    config = json.load(f)
sym='{}/{}-symbol.json'.format(params_path, net_name)
params='{}/{}-{:04d}.params'.format(params_path, net_name, epoch)
db_conf_p='../trained/2export/resnet_regressor.007/resnet_regressor.007_databases_alphs.json'
with open(db_conf_p, 'r') as f:
    db_config = json.load(f)
net = load_exported(sym, params, ctx)
net.hybridize()
#(b_size, 10, 243, 20)
print('config', config_p)
rna_alph = config['rna_exps_alphabet']
inference_shape = config['inference_shape']
max_label = float(config['denorm_max_label'])
prot_alph = config['protein_exps_alphabet']
databases_names_alph = config['databases']
writer_path = '{}/r18_tissue29_preds_extended.tsv'.format(params_path)
out_path = '{}/r18_tissue29_preds_labels_extended.txt'.format(params_path)
alph_to_write = ['geneId_uniprot', 'geneId_Ensembl'] + prot_alph
databases_data = []
for x in databases_names_alph:
    mtrx, alph = data_loader.mappingDatabase2matrix(x, db_config[x])
    if not mtrx.shape[1]:
        continue
    databases_data.append(mtrx)
print('writing to', writer_path)
print('process...')
with open(writer_path, 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(alph_to_write)
    genes_size = len(data_loader.genes())
    labels_outputs = []
    for j,gene in enumerate(data_loader.genes()):
        if j % 100 == 0:
            print('gene {} of {}'.format(j, genes_size))
        uid = gene.id()
        all_databases_gene_data = [x[j] for x in databases_data]
        sample = data_loader.gene2sampleExperimentHasId(
            uid,
            all_databases_gene_data,
            databases_names_alph,
            inference_shape[2],
            rna_alph,
            prot_alph 
        )
        if sample is None:
            print('error getting sample for', uid)
            continue
        ens = data_loader.uniprot2ensg(uid)
        if not len(ens):
            ens = ''
        else:
            ens = ens[0]
        def_v = [-1] * len(rna_alph)
        measurements = [uid, ens] + def_v
        for k in range(len(rna_alph)):
            if k >= len(sample):
                continue
            b = sample[k]
            batch2inf = mx.nd.array(b, ctx=ctx)
            batch2inf = batch2inf.expand_dims(axis=0)
            rna_expt_id = int(batch2inf[0][1].mean().asscalar())
            rna_expt = rna_alph[rna_expt_id]
            rna_value = gene.rna_measurements[rna_expt]
            prot_expt_id = int(batch2inf[0][2].mean().asscalar())
            prot_expt = prot_alph[prot_expt_id]
            prot_value = gene.protein_measurements[prot_expt]
            out = net(batch2inf)
            out_norm = denorm_shifted_log(out[0].asscalar()*max_label)
            labels_outputs.append([uid, rna_expt, rna_value, prot_expt, prot_value, out_norm])
            measurements[prot_expt_id+2] = out_norm
        writer.writerow(measurements)
    with open(out_path, 'w') as fv:
        for lo in labels_outputs:
            fv.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                lo[0],lo[1],
                lo[2],lo[3],
                lo[4],lo[5]
            ))
    print('data written to', out_path)
    print('tsv written to', writer_path)