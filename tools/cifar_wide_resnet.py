import os
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import cpu
from mxnet_dl.tools.tools import Softmax

def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

def _add_conv(out, channels=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(nn.BatchNorm(scale=True, momentum=0.99, epsilon=1e-3))
    if active:
        out.add(nn.Swish())

class MBConv(nn.HybridBlock):
    def __init__(self, channels, kernel, stride, **kwargs):
        super(MBConv, self).__init__(**kwargs)
        self.use_shortcut = stride == 1
        with self.name_scope():
            self.out = nn.HybridSequential()
            _add_conv(self.out, channels, active=True)
            _add_conv(self.out, channels, kernel=kernel, stride=stride,
                      pad=int((kernel-1)/2), num_group=1,
                      active=True)
            _add_conv(self.out, channels, active=False)

    def hybrid_forward(self, F, x):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out

class WideResNetBlock(HybridBlock):
    def __init__(self, channels, stride, downsample=False, drop_rate=0.0, in_channels=0, **kwargs):
        super(WideResNetBlock, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels, 1, channels)
        self.droprate = drop_rate
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        if self.droprate > 0:
            x = F.Dropout(x, self.droprate)
        x = self.conv2(x)
        return x + residual


class WideResNet(HybridBlock):
    def __init__(self, block, layers, channels, drop_rate, classes, **kwargs):
        super(WideResNet, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            # self.features_extractor = nn.HybridSequential(prefix='')
            # self.features_extractor.add(nn.BatchNorm(scale=False, center=False))
            # self.features_extractor.add(nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))
            # self.features_extractor.add(nn.BatchNorm())
            # in_channels = channels[0]
            # for i, num_layer in enumerate(layers):
            #     stride = 1 if i == 0 else 2
            #     self.features_extractor.add(self._make_layer(block, num_layer, channels[i+1], drop_rate,
            #                                        stride, i+1, in_channels=in_channels))
            #     in_channels = channels[i+1]
            self.features_extractor = WideResNet.get_fe(block, layers, channels, drop_rate)
            self.bn = nn.BatchNorm()
            self.features_to_dense = nn.HybridSequential(prefix='')
            self.features_to_dense.add(nn.GlobalAvgPool2D())
            self.features_to_dense.add(nn.Flatten())
            self.features_to_dense.add(nn.Dense(classes))
            self.output = Softmax()

    def hybrid_forward(self, F, x):
        x = self.features_extractor(x)
        x = F.relu(self.bn(x))
        x = self.features_to_dense(x)
        x = self.output(x)
        return x    

    @staticmethod
    def get_fe(block, layers, channels, drop_rate):
        features_extractor = nn.HybridSequential(prefix='')
        features_extractor.add(nn.BatchNorm(scale=False, center=False))
        features_extractor.add(nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))
        features_extractor.add(nn.BatchNorm())
        in_channels = channels[0]
        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            features_extractor.add(WideResNet._make_layer(block, num_layer, channels[i+1], drop_rate,
                                                stride, i+1, in_channels=in_channels))
            in_channels = channels[i+1]
        return features_extractor

    @staticmethod
    def _make_layer(block, layers, channels, drop_rate, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, drop_rate,
                            in_channels=in_channels, prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, drop_rate, in_channels=channels, prefix=''))
        return layer

def get_cifar_wide_resnet(num_layers, width_factor=1, drop_rate=0.0, ctx=cpu(), **kwargs):
    assert (num_layers - 4) % 6 == 0
    n = (num_layers - 4) // 6
    layers = [n] * 3
    channels = [16, 16*width_factor, 32*width_factor, 64*width_factor]
    net = WideResNet(WideResNetBlock, layers, channels, drop_rate, **kwargs)
    return net

class MResNet(HybridBlock):
    def __init__(self, block, layers, channels, drop_rate, classes, **kwargs):
        super(MResNet, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features_extractor = self._get_features_extractor(channels)
            _add_conv(self.features_extractor, 640, active=True)
            self.features_to_dense = nn.HybridSequential(prefix='')
            self.features_to_dense.add(nn.GlobalAvgPool2D())
            self.features_to_dense.add(nn.Conv2D(classes, 1, use_bias=False, prefix='pred_'))
            self.features_to_dense.add(nn.Flatten())
            self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else None
            self.output = Softmax()

    def hybrid_forward(self, F, x):
        x = self.features_extractor(x)
        x = self.features_to_dense(x)
        x = self.output(x)
        return x
    
    def _get_features_extractor(self, channels):
        features_extractor = nn.HybridSequential(prefix='')
        _add_conv(self.features, channels[0], kernel=3, stride=2, pad=1,
                        active=True)
        in_channels = channels[0]
        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            kernel = 3 if i % 3 != 0 else 5
            self.features_extractor.add(MBConv(channels=in_channels, kernel=kernel, stride=stride))
            in_channels = channels[i+1]
        return features_extractor

    def _make_layer(self, block, layers, channels, drop_rate, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, drop_rate,
                            in_channels=in_channels, prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, drop_rate, in_channels=channels, prefix=''))
        return layer