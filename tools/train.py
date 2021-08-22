import os
import numpy as np
import mxnet as mx
from data_loader import DataLoader
import argparse
from tools import str2bool
import yaml
import time
from cifar_wide_resnet import WideResNet, WideResNetBlock, get_cifar_wide_resnet
from mxnet.gluon.block import HybridBlock
from dense_model import DenseProteinPredectionNet
from mxnet.gluon import nn
from tools import (
    debug,
    str2bool, 
    readSearchXlsxReport, 
    roundUp,
    norm_shifted_log,
    denorm_shifted_log,
    list2nonEmptyIds,
    listfind
)
from gene_mapping import (
    mapping2dict, 
    uniprot_mapping_header, 
    uniq_nonempty_uniprot_mapping_header
)

isdebug = True

class RegressionResNet(HybridBlock):
    def __init__(self, block, layers, channels, drop_rate, **kwargs):
        super(RegressionResNet, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features_extractor = WideResNet.get_fe(block, layers, channels, drop_rate)
            self.bn = nn.BatchNorm()
            self.features_to_dense = nn.HybridSequential(prefix='')
            self.features_to_dense.add(nn.GlobalAvgPool2D())
            self.features_to_dense.add(nn.Flatten())
            self.features_to_dense.add(nn.Dense(1))
    
        def hybrid_forward(self, F, x):
            x = self.features_extractor(x)
            x = F.relu(self.bn(x))
            x = self.features_to_dense(x)
            return x

def get_regression_wide_resnet(
        num_layers, 
        width_factor=1, 
        drop_rate=0.0, 
        ctx=mx.cpu(), 
        **kwargs
    ):
    assert (num_layers - 4) % 6 == 0
    n = (num_layers - 4) // 6
    layers = [n] * 3
    channels = [16, 16*width_factor, 32*width_factor, 64*width_factor]
    net = RegressionResNet(WideResNetBlock, layers, channels, drop_rate, **kwargs)
    return net

class ProteinAbundanceTrainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.load(f)
        self.data_loader = DataLoader(config_path)
        try:
            self.ctx = [mx.gpu()]
            a = mx.nd.array([[0]], ctx=self.ctx[0])
        except:
            self.ctx = [mx.cpu()]
        self.train_settings = self.config['train']
        self.lr_settings = self.train_settings['lr']
        self.batch_settings = self.train_settings['batch_size']
        self.augm_settings = self.train_settings['augm']
        self.with_augm = self.augm_settings['isEnabled']
        self.epochs = self.train_settings['epochs']
        self.log_interval = self.train_settings['log_interval']
        self.wd = self.train_settings['wd']
        self.momentum = self.train_settings['momentum']
        self.optimizer = self.train_settings['optimizer'] 
        num_layers=16
        width_factor=2.0
        assert (num_layers - 4) % 6 == 0
        n = (num_layers - 4) // 6
        layers = [n] * 3
        channels = [16, 16*width_factor, 32*width_factor, 64*width_factor]
        drop_rate = 0.0
        # self.model = WideResNet.get_fe(WideResNetBlock, layers, channels, drop_rate)
        # self.model = get_cifar_wide_resnet(16, 1, classes=['asd', 'lal'], ctx=mx.cpu())
        # self.model = DenseProteinPredectionNet(80)
        # exit(0)
        self.model = get_regression_wide_resnet(16, 1, drop_rate=0.0)
        print(self.model.features_extractor)
        exit(0)

        
    def train_loop(self):
        lrs = [v for _,v in self.lr_settings.items()]
        # print(lrs)
        self.model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx[0])
        self.model.collect_params().reset_ctx(self.ctx[0])
        self.model.hybridize()
        optimizer_params = {
            'wd': self.wd, 
            'momentum': self.momentum, 
            'learning_rate': lrs[0]
        }
        trainer = mx.gluon.Trainer(
            self.model.collect_params(), 
            self.optimizer, 
            optimizer_params
        )
        max_measures = self.data_loader.maxRnaMeasurementsInData()
        databases = uniq_nonempty_uniprot_mapping_header()
        if isdebug:
            databases = databases[:2]
        databases_data = []
        databases2use =[]
        for x in databases:
            mtrx = self.data_loader.mappingDatabase2matrix(x)
            if not mtrx.shape[1]:
                continue
            databases_data.append(mtrx)
            databases2use.append(x)
        genes_exps_batches = []
        for j, gene in enumerate(self.data_loader.genes):
            if isdebug:
                if j >= 10:
                    break
            print('gene {} of {}'.format(j, len(self.data_loader.genes)))
            all_databases_gene_data = [x[j] for x in databases_data]
            genes_exps_batches.append(
                self.data_loader.gene2sampleExperimentHasId(
                    gene.id_uniprot, 
                    all_databases_gene_data,
                    databases2use,
                    max_measures
                )
            )
            break
        for i in range(self.epochs):
            tic = time.time()
            for j,gene_experiments in enumerate(genes_exps_batches):
                for exp in gene_experiments:
                    mxarr = mx.nd.array(exp)
                    # mxarr = mx.nd.zeros(shape=(3,64,64))
                    mxarr = mxarr.expand_dims(0)
                    debug(mxarr.shape)
                    out = self.model(mxarr)
                    debug(out)
                    debug(out.shape)
                    exit()
            epoch_time = time.time() - tic
            print('[Epoch: {}] time: {}'.format(i, epoch_time))
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    opt = parser.parse_args()
    trainer = ProteinAbundanceTrainer(opt.config)
    trainer.data_loader.info()
    trainer.train_loop()
    
    
    
    

