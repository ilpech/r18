import os
import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class DenseProteinPredectionNet(HybridBlock):
    def __init__(self, channels, drop_rate=0.0, **kwargs):
        super(DenseProteinPredectionNet, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.dense1 = nn.Dense(channels, 'relu')
        self.bn2 = nn.BatchNorm()
        self.dense2 = nn.Dense(channels, 'relu')
        self.bn3 = nn.BatchNorm()
        self.dense3 = nn.Dense(channels, 'relu')
        self.droprate = drop_rate
        self.features_to_addWeight = nn.HybridSequential(prefix='')
        self.features_to_addWeight.add(nn.GlobalAvgPool2D())
        self.features_to_addWeight.add(nn.Flatten())
        self.features_to_addWeight.add(nn.Dense(1))
        self.features_to_mulWeight = nn.HybridSequential(prefix='')
        self.features_to_mulWeight.add(nn.GlobalAvgPool2D())
        self.features_to_mulWeight.add(nn.Flatten())
        self.features_to_mulWeight.add(nn.Dense(1))
        self.out = nn.Dense(1)

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = self.dense1(x)
        x = F.relu(self.bn2(x))
        if self.droprate > 0:
            x = F.Dropout(x, self.droprate)
        x = self.dense2(x)
        x = F.relu(self.bn3(x))
        x = self.dense3(x)
        # x = self.features_to_mulWeight(x)
        x = self.out(x)
        return x
      
if __name__ == '__main__':
    ctx = mx.cpu()
    model = DenseProteinPredectionNet(80)
    model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    model.collect_params().reset_ctx(ctx)
    data = mx.nd.array([[0.8, 1, 1, 0, 0, 0, 0, 1]])
    print(data)
    # exit()
    print(model(data))
