import os
from glob import glob 

import torch
import numpy as np

default_dtype_torch = torch.float64
default_torch_device = torch.device('cpu')


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
