import os
import numpy as np
import mxnet as mx
import shutil
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

# isdebug = True
isdebug = False

class TrainLogger:
    def __init__(
        self, 
        outpath
    ):
        self.out = outpath
        if os.path.isfile(self.out):
            with open(self.out, 'w') as f:
                pass
        else:
            if not os.path.isdir(os.path.split(self.out)[0]):
                os.makedirs(os.path.split(self.out)[0])
            with open(self.out, 'w') as f:
                pass
        print('log file is opened at', self.out)
    
    def write(self, m):
        try:
            with open(self.out, 'a') as f:
                f.write(m)
        except:
            pass
    
    def print(self, m, show=True, write=True):
        if show:
            print(m)
        if write:
            self.write(m+'\n')

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
        self.train_settings = self.config['train']
        self.net_name = self.train_settings['net_name']
        self.params_path = '../trained/{}'.format(self.net_name)
        self.log_path = '{}/{}_log.txt'.format(self.params_path, self.net_name)
        self.logger = TrainLogger(self.log_path)
        self.data_loader = DataLoader(config_path)
        self.data_loader.loadTissue29data2genes(
            '../data/liver_hepg2/tissue29.05k_rna.tsv',
            '../data/liver_hepg2/tissue29.05k_prot.tsv',
            create_new_genes=True,
            isdebug=isdebug
        )
        try:
            self.ctx = [mx.gpu()]
            a = mx.nd.array([[0]], ctx=self.ctx[0])
            self.logger.print('successfully created gpu array -> using gpu')
        except:
            self.ctx = [mx.cpu()]
            self.logger.print('cannot create gpu array -> using cpu')
        self.lr_settings = self.train_settings['lr']
        self.batch_settings = self.train_settings['batch_size']
        self.augm_settings = self.train_settings['augm']
        self.with_augm = self.augm_settings['isEnabled']
        self.epochs = self.train_settings['epochs']
        self.log_interval = self.train_settings['log_interval']
        self.wd = self.train_settings['wd']
        self.momentum = self.train_settings['momentum']
        self.optimizer = self.train_settings['optimizer']
        drop_rate = 0.0
        num_layers=22
        width_factor=2
        if isdebug:
            num_layers=16
            width_factor=1
        assert (num_layers - 4) % 6 == 0
        self.logger.print('creating resnset regression model...')
        self.model = RegressionResNet.get_regression_wide_resnet(
            num_layers, 
            width_factor, 
            drop_rate=0.0
        )
        self.logger.print('created. layers={} wf={}'.format(num_layers, width_factor))

    def save_checkpoint(self, params_path, epoch):
        cwd = os.getcwd()
        os.chdir(params_path)
        export_name = '{}_{:03d}.params'.format(self.net_name, epoch)
        self.model.save_parameters(export_name)
        # !realize
        # self.write_config('.')
        os.chdir(cwd)
    
    def export_nn(self, epoch, export_path=None):
        if export_path == None:
            export_path = os.path.join(self.params_path, '..')
        exp_dir = os.path.join(export_path, self.net_name)
        cwd = os.getcwd()
        if not os.path.isdir(exp_dir):
            ensure_folder(exp_dir)
        os.chdir(exp_dir)
        self.model.export(self.net_name, epoch=epoch)
        self.weights_path = '{}-{:04d}.params'.format(self.net_name, epoch)
        self.sym_path = '{}-symbol.json'.format(self.net_name)
        self.logger.print('Network was successfully exported at {}'.format(exp_dir))
        os.chdir(cwd)

    def load_exported(self, sym_path='', params_path=''):
        """
        load model
        self.sym_path and self.params_path are using if given are not exist
        """
        if not os.path.isfile(params_path):
            self.logger.print(self.weights_path)
            def_w_p = os.path.join(self.params_path, self.weights_path)
            if os.path.isfile(os.path.join(self.params_path, self.weights_path)):
                params_path = os.path.join(self.params_path, self.weights_path)
                self.logger.print('params path from config used')
            else:
                self.logger.print('Check params path', params_path)
                raise FileNotFoundError(params_path)
        if not os.path.isfile(sym_path):
            self.logger.print('Check sym path', sym_path)
            if os.path.isfile(os.path.join(self.params_path, self.sym_path)):
                sym_path = os.path.join(self.params_path, self.sym_path)
                self.logger.print('sym path from config used')
            else:
                self.logger.print('Check sym path', sym_path)
                raise FileNotFoundError(sym_path)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.net = mx.gluon.nn.SymbolBlock.imports(
                sym_path, 
                ['data'], 
                params_path, 
                ctx=self.ctx[0]
            )
            self.sym_path = sym_path
            self.params_path = params_path
        

    def train_loop(self):
        lrs = [v for _,v in self.lr_settings.items()]
        self.model.collect_params().initialize(
            mx.init.Xavier(magnitude=2.24), 
            ctx=self.ctx[0]
        )
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
        databases = uniq_nonempty_uniprot_mapping_header()
        data, labels = self.data_loader.data(isdebug)
        data_cnt = len(labels)
        data2val_cnt = roundUp(data_cnt/5)
        max_label = np.max(labels)
        norm_labels = [float(x/max_label) for x in labels]
        data2train = mx.gluon.data.dataset.ArrayDataset(data[data2val_cnt:], norm_labels[data2val_cnt:])
        data2val = mx.gluon.data.dataset.ArrayDataset(data[:data2val_cnt], norm_labels[:data2val_cnt])
        batch_size = 32
        if isdebug:
            batch_size = 3
        debug(data_cnt)
        debug(len(data2train))
        debug(len(data2val))
        train_loader = mx.gluon.data.DataLoader(data2train, batch_size=batch_size, shuffle=True)
        val_loader = mx.gluon.data.DataLoader(data2val, batch_size=batch_size, shuffle=False)
        L = mx.gluon.loss.L2Loss()
        num_batch = roundUp(len(labels)/batch_size)
        train_metric = mx.metric.MSE()
        train_denorm_metric = mx.metric.MSE()
        val_metric = mx.metric.MSE()
        val_denorm_metric = mx.metric.MSE()
        best_epoch = 0
        min_val_error = None
        first_forward = True
        for i in range(1, self.epochs):
            # labels, shuffle_indxs = shuffle(labels)
            # data = setindxs(data, shuffle_indxs)
            tic = time.time()
            train_metric.reset()
            val_metric.reset()
            train_denorm_metric.reset()
            val_denorm_metric.reset()
            train_loss = 0
            passed = 0
            while passed < data_cnt:
                for data, labels in train_loader:
                    if first_forward:
                        self.logger.print('batch shape::{}'.format(data.shape))
                        first_forward = False
                    ctx_data = data.as_in_context(self.ctx[0]).astype('float32')
                    with mx.autograd.record():
                        out = self.model(ctx_data)
                        loss = L(
                            out, 
                            labels.astype('float32').as_in_context(self.ctx[0])
                        )
                        loss.backward()
                        passed += len(out)
                    trainer.step(len(data))
                    train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
                    train_metric.update(labels, out)
                    denorm_labels = mx.nd.array([denorm_shifted_log(x.asscalar()) for x in labels])
                    denorm_out = mx.nd.array([denorm_shifted_log(x.asscalar()) for x in out])
                    train_denorm_metric.update(denorm_labels, denorm_out)
            for data, labels in val_loader:
                ctx_data = data.as_in_context(self.ctx[0]).astype('float32')
                out = self.model(ctx_data)
                val_metric.update(labels, out)
                denorm_labels = mx.nd.array([denorm_shifted_log(x.asscalar()) for x in labels])
                denorm_out = mx.nd.array([denorm_shifted_log(x.asscalar()) for x in out])
                val_denorm_metric.update(denorm_labels, denorm_out)
            _, train_denorm_acc = train_denorm_metric.get()
            _, val_denorm_acc = val_denorm_metric.get()
            _, train_acc = train_metric.get()
            _, val_acc = val_metric.get()
            new_best_val = False
            if not min_val_error:
                min_val_error = val_acc
            else:
                if abs(val_acc) < min_val_error:
                    min_val_error = val_acc
                    new_best_val = True
                    best_epoch = i
            train_loss /= num_batch
            epoch_time = time.time() - tic
            msg = '[Epoch::{:03d}] time::{:.1f} \n'\
                  '| (val)::MSE_metric::{:.8f} \n'\
                  '| (val)::MSE_denorm_metric::{:.8f} \n'\
                  '| (train)::MSE_metric::{:.8f} \n'\
                  '| (train)::MSE_denorm_metric::{:.8f} \n'\
                  '| (train)::MES_loss::{:.8f}'.format(
                  i, 
                  epoch_time,
                  val_acc,
                  val_denorm_acc,
                  train_acc,
                  train_denorm_acc,
                  train_loss
            )
            self.logger.print(msg)
            if not i % 10 or (i > 30 and new_best_val):
                if new_best_val:
                    self.logger.print('new best val')
                if not isdebug:
                    self.export_nn(i, '../trained')
            if min_val_error and not new_best_val:
                self.logger.print('best val was at epoch({})::{:.8f}'.format(best_epoch, min_val_error))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    opt = parser.parse_args()
    trainer = ProteinAbundanceTrainer(opt.config)
    # trainer.data_loader.info()
    trainer.train_loop()
    
    
    
    

