import math
import torch
#import torch.nn as nn
import architectures.hybrid_modules as nn
from architectures import utils
import torch.nn.functional as F


class ResBlock(nn.HybridModule):
    def __init__(self, nFin, nFout):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential()
        self.conv_block.add_module('BNorm1', nn.BatchNorm2d(nFin))
        self.conv_block.add_module('LRelu1', nn.LeakyReLU(0.1))
        self.conv_block.add_module('ConvL1',
            nn.Conv2d(nFin,  nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm2', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu2', nn.LeakyReLU(0.1))
        self.conv_block.add_module('ConvL2',
            nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm3', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu3', nn.LeakyReLU(0.1))
        self.conv_block.add_module('ConvL3',
            nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))

        self.skip_layer = nn.Conv2d(nFin, nFout, kernel_size=1, stride=1)

    def forward(self, x, params=None, buffers=None, bn_training=None, do_training=None, no_grad_hint_mask=None):
        if params is None:
            params = [p for p in self.parameters()]
        if buffers is None:
            buffers = [p for p in utils.get_buffers(self)]
        if no_grad_hint_mask is None:
            no_grad_hint_mask = '0' * len([m for m in self.modules()])

        if bn_training is None:
            bn_training = [self.training] * self.get_num_batchnorms()
        if do_training is None:
            do_training = self.training

        zipped = list(zip(self.children(), self.unmerge_parameters(params), self.unmerge_buffers(buffers), self.unmerge_bn_training(bn_training), self.unmerge_no_grad_hint_mask(no_grad_hint_mask)))
        assert len(zipped) == 2

        b, p, pb, bt, nghm = zipped[0]
        assert b is self.conv_block
        conv_block_result = self.conv_block(x, params=p, buffers=pb, bn_training=bt, do_training=do_training, no_grad_hint_mask=nghm)
        b, p, pb, bt, nghm = zipped[1]
        assert b is self.skip_layer
        skip_layer_result = self.skip_layer(x, params=p, buffers=pb, bn_training=bt, do_training=do_training, no_grad_hint_mask=nghm)
        return skip_layer_result + conv_block_result


class ResNetLike(nn.HybridModule):
    def __init__(self, opt):
        super(ResNetLike, self).__init__()
        self.in_planes = opt['in_planes']
        self.out_planes = [64, 96, 128, 256]
        self.num_stages = 4

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert(type(self.out_planes)==list)
        assert(len(self.out_planes)==self.num_stages)
        num_planes = [self.out_planes[0],] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else False
        dropout = opt['dropout'] if ('dropout' in opt) else 0

        self.feat_extractor = nn.Sequential()
        self.feat_extractor.add_module('ConvL0',
            nn.Conv2d(self.in_planes, num_planes[0], kernel_size=3, padding=1))
        for i in range(self.num_stages):
            self.feat_extractor.add_module('ResBlock'+str(i),
                ResBlock(num_planes[i], num_planes[i+1]))
            self.feat_extractor.add_module('MaxPool'+str(i),
                nn.MaxPool2d(kernel_size=2,stride=2,padding=0))

        self.feat_extractor.add_module('AvgPool', nn.AdaptiveAvgPool2d(1))
        self.feat_extractor.add_module('BNormF1',
            nn.BatchNorm2d(num_planes[-1]))
        self.feat_extractor.add_module('ReluF1', nn.ReLU(inplace=True))
        self.feat_extractor.add_module('ConvLF1',
            nn.Conv2d(num_planes[-1], 384, kernel_size=1))
        if dropout>0.0:
            self.feat_extractor.add_module('DropoutF1',
                nn.Dropout(p=dropout, inplace=False))

        self.feat_extractor.add_module('BNormF2', nn.BatchNorm2d(384))
        self.feat_extractor.add_module('ReluF2', nn.ReLU(inplace=True))
        self.feat_extractor.add_module('ConvLF2',
            nn.Conv2d(384, 512, kernel_size=1))
        self.feat_extractor.add_module('BNormF3', nn.BatchNorm2d(512))
        if dropout>0.0:
            self.feat_extractor.add_module('DropoutF2',
                nn.Dropout(p=dropout, inplace=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, params=None, buffers=None, bn_training=None, do_training=None, no_grad_hint_mask=None):
        if params is None:
            params = [p for p in self.parameters()]
        if buffers is None:
            buffers = [p for p in utils.get_buffers(self)]
        if no_grad_hint_mask is None:
            no_grad_hint_mask = '0' * len([m for m in self.modules()])

        if bn_training is None:
            bn_training = [self.training] * self.get_num_batchnorms()
        if do_training is None:
            do_training = self.training

        out = self.feat_extractor(x, params=params, buffers=buffers, bn_training=bn_training, do_training=do_training, no_grad_hint_mask=no_grad_hint_mask[1:])
        return out.view(out.size(0),-1)


def create_model(opt):
    return ResNetLike(opt)
