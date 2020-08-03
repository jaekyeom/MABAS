import collections
from itertools import repeat

import torch

# Backward-compatible buffer getters
def get_named_buffers(this, memo=None, prefix=''):
    if hasattr(this, 'named_buffers'):
        for v in this.named_buffers():
            yield v

    if memo is None:
        memo = set()
    for name, p in this._buffers.items():
        if p is not None and p not in memo:
            memo.add(p)
            yield prefix + ('.' if prefix else '') + name, p
    for mname, module in this.named_children():
        submodule_prefix = prefix + ('.' if prefix else '') + mname
        for name, p in get_named_buffers(module, memo, submodule_prefix):
            yield name, p
def get_buffers(this):
    if hasattr(this, 'buffers'):
        return this.buffers()
    return this._all_buffers()

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

