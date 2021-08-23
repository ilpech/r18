import os
import numpy as np
import mxnet as mx
from data_loader import DataLoader
import argparse
from tools import str2bool
import yaml
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
    listfind
)
from gene_mapping import (
    mapping2dict, 
    uniprot_mapping_header, 
    uniq_nonempty_uniprot_mapping_header
)

isdebug = False

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
            self.out = nn.Dense(1)
    
    def hybrid_forward(self, F, x):
        x = self.features_extractor(x)
        x = F.relu(self.bn(x))
        x = self.features_to_dense(x)
        x = self.out(x)
        return x
    
    @staticmethod
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

    @staticmethod
    def checknet():
        net = RegressionResNet.get_regression_wide_resnet(16, 2, 0.0, mx.cpu())
        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=mx.cpu())
        net.collect_params().reset_ctx(mx.cpu())
        b = mx.nd.zeros(shape=(1,3,64,64))
        debug(b.shape)
        debug(net.features_extractor(b).shape)
        debug(net.features_to_dense(b).shape)
        debug(net(b).shape)
        a = mx.nd.random.uniform(shape=(2,18,609,20))
        c = mx.nd.random.uniform(shape=(2,18,300,20))
        d = mx.nd.random.uniform(shape=(2,18,300,22))
        net = RegressionResNet.get_regression_wide_resnet(16, 2, 0.0, mx.cpu())
        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=mx.cpu())
        net.collect_params().reset_ctx(mx.cpu())
        debug(a.shape)
        debug(net(a).shape)
        debug(net(a))
        debug(c.shape)
        debug(net(c).shape)
        debug(net(c))
        debug(d.shape)
        debug(net(d).shape)
        debug(net(d))
        # but cannot do net(b) because of 3 channels

class ProteinAbundanceTrainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.load(f)
        self.data_loader = DataLoader(config_path)
        try:
            self.ctx = [mx.gpu()]
            a = mx.nd.array([[0]], ctx=self.ctx[0])
            print('successfully created gpu array -> using gpu')
        except:
            self.ctx = [mx.cpu()]
            print('cannot create gpu array -> using cpu')
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
        self.model = RegressionResNet.get_regression_wide_resnet(
            16, 
            1, 
            drop_rate=0.0
        )

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
            databases = databases[:3]
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
                if j >= 3:
                    print('debug::genes::', j)
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
        data = []
        labels = []
        for gene_id in range(len(genes_exps_batches)):
            gene = self.data_loader.genes[gene_id] # проверить точно ли правильная индексация?
            for exp_id in range(len(genes_exps_batches[gene_id])):
                try:
                    data.append(genes_exps_batches[gene_id][exp_id].astype('float32'))
                    labels.append(gene.protein_copies_per_cell_1D)
                except:
                    pass
        data_cnt = len(labels)
        max_label = np.max(labels)
        norm_labels = [float(x/max_label) for x in labels]
        data_batch = mx.gluon.data.dataset.ArrayDataset(data, norm_labels)
        batch_size = 5
        train_loader = mx.gluon.data.DataLoader(data_batch, batch_size=batch_size, shuffle=True)
        genes_exps_batches = mx.nd.array(genes_exps_batches)
        L = mx.gluon.loss.L2Loss()
        num_batch = roundUp(len(labels)/batch_size)
        train_metric = mx.metric.MSE()
        for i in range(self.epochs):
            tic = time.time()
            train_metric.reset()
            train_loss = 0
            passed = 0
            while passed < len(data_batch):
                for data, labels in train_loader:
                    ctx_data = data.as_in_context(self.ctx[0]).astype('float32')
                    with mx.autograd.record():
                        out = self.model(ctx_data)
                        loss = L(out, labels.astype('float32').as_in_context(self.ctx[0]))
                        loss.backward()
                        passed += len(out)
                    trainer.step(len(data))
                    train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
                    train_metric.update(labels, out)
            _, train_acc = train_metric.get()
            train_loss /= num_batch
            epoch_time = time.time() - tic
            print('[Epoch::{:03d}] time::{:.1f} | MSE_metric::{:.4f} | MES_loss::{:.4f}'.format(
                i, 
                epoch_time,
                train_acc,
                train_loss
            ))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    opt = parser.parse_args()
    trainer = ProteinAbundanceTrainer(opt.config)
    # trainer.data_loader.info()
    trainer.train_loop()
    
    
    
    

