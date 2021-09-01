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

# isdebug = True
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
