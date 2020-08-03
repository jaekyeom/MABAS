from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.utils import _single, _pair, _triple, get_buffers

class NoopContext(object):
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

class HybridModule(nn.Module):
    def __init__(self):
        super(HybridModule, self).__init__()
        self._keys_params = list()
        self._keys_buffers = list()
        self._index_params = dict()
        self._index_buffers = dict()

    def reset_parameters(self):
        raise NotImplementedError

    def add_module(self, name, module):
        assert isinstance(module, HybridModule)
        return super(HybridModule, self).add_module(name, module)

    def add_hybrid_param(self, key, val):
        assert (key not in self._index_params)
        assert (val.requires_grad)
        self._index_params[key] = len(self._keys_params)
        self._keys_params.append(key)
        self.register_parameter(key, val)
        return val

    def add_hybrid_buffer(self, key, val):
        assert (key not in self._index_buffers)
        #assert (not val.requires_grad)
        self._index_buffers[key] = len(self._keys_buffers)
        self._keys_buffers.append(key)
        self.register_buffer(key, val)
        return val

    def get_hybrid_param(self, params, key):
        return params[self._index_params[key]]

    def get_hybrid_buffer(self, buffers, key):
        return buffers[self._index_buffers[key]]

    def has_hybrid_param(self, key):
        return key in self._index_params

    def has_hybrid_buffer(self, key):
        return key in self._index_buffers

    def get_num_batchnorms(self):
        return len([m for m in self.modules() if isinstance(m, BatchNorm)])

    def unmerge_parameters(self, params):
        children = self.children()
        if params is None:
            return [None] * len([b for b in children])
        params = [p for p in params]
        nums_params = [len([p for p in b.parameters()]) for b in children]
        assert(sum(nums_params) == len(params))
        unmerged_params = []
        for n in nums_params:
            unmerged_params.append(params[:n])
            params = params[n:]
        return unmerged_params

    def unmerge_buffers(self, buffers):
        children = self.children()
        if buffers is None:
            return [None] * len([b for b in children])
        buffers = [p for p in buffers]
        nums_buffers = [len([p for p in get_buffers(b)]) for b in children]
        assert(sum(nums_buffers) == len(buffers))
        unmerged_buffers = []
        for n in nums_buffers:
            unmerged_buffers.append(buffers[:n])
            buffers = buffers[n:]
        return unmerged_buffers

    def unmerge_bn_training(self, bn_training):
        children = self.children()
        if bn_training is None:
            return [None] * len([b for b in children])
        nums_bn = [b.get_num_batchnorms() for b in children]
        assert(sum(nums_bn) == len(bn_training))
        unmerged = []
        for n in nums_bn:
            unmerged.append(bn_training[:n])
            bn_training = bn_training[n:]
        return unmerged

    def unmerge_no_grad_hint_mask(self, mask):
        children = self.children()
        if mask is None:
            return [None] * len([b for b in children])
        if mask[0] != '0':
            assert all([(c != '0') for c in mask])
        mask = mask[1:]
        nums_modules = [len([p for p in b.modules()]) for b in children]
        assert(sum(nums_modules) == len(mask))
        unmerged_masks = []
        for n in nums_modules:
            unmerged_masks.append(mask[:n])
            mask = mask[n:]
        return unmerged_masks


class BatchNorm(HybridModule):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNorm, self).__init__()

        if not affine:
            raise NotImplementedError
        if momentum is None:
            raise NotImplementedError

        self.eps = eps
        self.momentum = momentum

        p = self.add_hybrid_param('weight', nn.Parameter(torch.ones(num_features)))
        p.data.uniform_()
        p = self.add_hybrid_param('bias', nn.Parameter(torch.zeros(num_features)))
        p.data.zero_()
        p = self.add_hybrid_buffer('running_mean',
                torch.zeros(num_features))
        p.zero_()
        p = self.add_hybrid_buffer('running_var',
                torch.ones(num_features))
        p.fill_(1)

        # No num_batches_tracked here for backward compatibility

    def forward(self, x_input, params=None, buffers=None, bn_training=None, do_training=None, no_grad_hint_mask=None):
        if params is None:
            params = [p for p in self.parameters()]
        if buffers is None:
            buffers = [p for p in get_buffers(self)]
        if no_grad_hint_mask is None:
            no_grad_hint_mask = '0' * len([m for m in self.modules()])

        if bn_training is None:
            bn_training = [self.training] * self.get_num_batchnorms()
        if do_training is None:
            do_training = self.training

        assert len(no_grad_hint_mask) == 1
        with (NoopContext if no_grad_hint_mask == '0' else torch.no_grad)():
            x = F.batch_norm(
                    x_input,
                    self.get_hybrid_buffer(buffers, 'running_mean'),
                    self.get_hybrid_buffer(buffers, 'running_var'),
                    weight=self.get_hybrid_param(params, 'weight'),
                    bias=self.get_hybrid_param(params, 'bias'),
                    training=bn_training.pop(0),
                    momentum=self.momentum,
                    eps=self.eps)

        assert len(bn_training) == 0
        return x

BatchNorm2d = BatchNorm


class Conv2d(HybridModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups

        n = in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        p = self.add_hybrid_param('weight', nn.Parameter(torch.ones(out_channels, in_channels // self.groups, *self.kernel_size)))
        p.data.uniform_(-stdv, stdv)
        if bias:
            p = self.add_hybrid_param('bias', nn.Parameter(torch.zeros(out_channels)))
            p.data.uniform_(-stdv, stdv)

    def forward(self, x_input, params=None, buffers=None, bn_training=None, do_training=None, no_grad_hint_mask=None):
        if params is None:
            params = [p for p in self.parameters()]
        if buffers is None:
            buffers = [p for p in get_buffers(self)]
        if no_grad_hint_mask is None:
            no_grad_hint_mask = '0' * len([m for m in self.modules()])

        if bn_training is None:
            bn_training = [self.training] * self.get_num_batchnorms()
        if do_training is None:
            do_training = self.training

        assert len(no_grad_hint_mask) == 1
        with (NoopContext if no_grad_hint_mask == '0' else torch.no_grad)():
            x = F.conv2d(
                    x_input,
                    self.get_hybrid_param(params, 'weight'),
                    bias=(self.get_hybrid_param(params, 'bias') if self.has_hybrid_param('bias') else None),
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups)

        assert len(bn_training) == 0
        return x


class ReLU(HybridModule):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x_input, params=None, buffers=None, bn_training=None, do_training=None, no_grad_hint_mask=None):
        if no_grad_hint_mask is None:
            no_grad_hint_mask = '0' * len([m for m in self.modules()])

        assert len(no_grad_hint_mask) == 1
        with (NoopContext if no_grad_hint_mask == '0' else torch.no_grad)():
            x = F.relu(x_input, inplace=self.inplace)
        return x

class LeakyReLU(HybridModule):
    def __init__(self, negative_slope=1e-2, inplace=False):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x_input, params=None, buffers=None, bn_training=None, do_training=None, no_grad_hint_mask=None):
        if no_grad_hint_mask is None:
            no_grad_hint_mask = '0' * len([m for m in self.modules()])

        assert len(no_grad_hint_mask) == 1
        with (NoopContext if no_grad_hint_mask == '0' else torch.no_grad)():
            x = F.leaky_relu(x_input, negative_slope=self.negative_slope, inplace=self.inplace)
        return x

class MaxPool2d(HybridModule):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x_input, params=None, buffers=None, bn_training=None, do_training=None, no_grad_hint_mask=None):
        if no_grad_hint_mask is None:
            no_grad_hint_mask = '0' * len([m for m in self.modules()])

        assert len(no_grad_hint_mask) == 1
        with (NoopContext if no_grad_hint_mask == '0' else torch.no_grad)():
            x = F.max_pool2d(
                    x_input, self.kernel_size, self.stride,
                    self.padding, self.dilation, self.ceil_mode,
                    self.return_indices)
        return x

class AvgPool2d(HybridModule):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x_input, params=None, buffers=None, bn_training=None, do_training=None, no_grad_hint_mask=None):
        if no_grad_hint_mask is None:
            no_grad_hint_mask = '0' * len([m for m in self.modules()])

        assert len(no_grad_hint_mask) == 1
        with (NoopContext if no_grad_hint_mask == '0' else torch.no_grad)():
            return F.avg_pool2d(input, self.kernel_size, self.stride,
                                self.padding, self.ceil_mode, self.count_include_pad)



class AdaptiveAvgPool2d(HybridModule):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x_input, params=None, buffers=None, bn_training=None, do_training=None, no_grad_hint_mask=None):
        if no_grad_hint_mask is None:
            no_grad_hint_mask = '0' * len([m for m in self.modules()])

        assert len(no_grad_hint_mask) == 1
        with (NoopContext if no_grad_hint_mask == '0' else torch.no_grad)():
            x = F.adaptive_avg_pool2d(x_input, self.output_size)
        return x

class Dropout(HybridModule):
    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, x_input, params=None, buffers=None, bn_training=None, do_training=None, no_grad_hint_mask=None):
        if no_grad_hint_mask is None:
            no_grad_hint_mask = '0' * len([m for m in self.modules()])

        if do_training is None:
            do_training = self.training

        assert len(no_grad_hint_mask) == 1
        with (NoopContext if no_grad_hint_mask == '0' else torch.no_grad)():
            x = F.dropout(x_input, self.p, do_training, self.inplace)
        return x

class Sequential(HybridModule):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __len__(self):
        return len(self._modules)

    def forward(self, x_input, params=None, buffers=None, bn_training=None, do_training=None, no_grad_hint_mask=None):
        if params is None:
            params = [p for p in self.parameters()]
        if buffers is None:
            buffers = [p for p in get_buffers(self)]
        if no_grad_hint_mask is None:
            no_grad_hint_mask = '0' * len([m for m in self.modules()])

        if bn_training is None:
            bn_training = [self.training] * self.get_num_batchnorms()
        if do_training is None:
            do_training = self.training

        zipped = list(zip(self.children(), self.unmerge_parameters(params), self.unmerge_buffers(buffers), self.unmerge_bn_training(bn_training), self.unmerge_no_grad_hint_mask(no_grad_hint_mask)))

        x = x_input
        for b, p, pb, bt, nghm in zipped:
            x = b(x, params=p, buffers=pb, bn_training=bt, do_training=do_training, no_grad_hint_mask=nghm)

        return x

