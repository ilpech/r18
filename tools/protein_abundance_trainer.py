import os
import numpy as np
from dense_model import DenseProteinPredectionNet
import mxnet as mx
from data_loader import DataLoader
import argparse
from tools import str2bool
import yaml
import time

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
        self.model = DenseProteinPredectionNet(80)
        
    def train_loop(self):
        lrs = [v for _,v in self.lr_settings.items()]
        # print(lrs)
        self.model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx[0])
        self.model.collect_params().reset_ctx(self.ctx[0])
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
        for i in range(self.epochs):
            tic = time.time()
            epoch_time = time.time() - tic
            print('[Epoch: {}] time: {}'.format(i, epoch_time))
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    opt = parser.parse_args()
    trainer = DenseTrainer(opt.config)
    trainer.data_loader.info()
    trainer.train_loop()
    
    
    
    

