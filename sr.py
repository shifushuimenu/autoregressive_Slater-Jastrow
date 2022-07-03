"""Stochastic reconfiguration

References:
===========
[1] S. Sorella, Green Function Monte Carlo with Stochastic Recon-
    figuration, Phys. Rev. Lett. 80, 4558 (1998).
[2] S. Sorella, Wave function optimization in the variational Monte
    Carlo method, Phys. Rev. B 71, 241103(R) (2005).
[3] S.-I. Amari, Natural gradient works efficiently in learning,
    Neural Comput. 10, 251 (1998).
"""

import numpy as np
import torch.nn as nn

# TODO: introduce regularization parameter before inverting S

class SR(object):
    def  __init__(self, num_params, l_reg=1e-8):
        self.num_params = num_params 
        assert l_reg > 0 
        self.l_reg = l_reg
        self.Fisher_matrix = np.zeros((num_params, num_params), dtype=np.float32)
        self.av_grad = np.zeros((num_params,1), dtype=np.float32)
        self.counter = 0

    def acc_Fisher_infomatrix(self, grad_loc):
        """
        """
        self.counter += 1 
        self.Fisher_matrix += np.outer(grad_loc, grad_loc)
        self.av_grad += grad_loc

    def av_Fisher_infomatrix(self):
        self.Fisher_matrix /= self.counter 
        self.av_grad /= self.counter 

        self.Fisher_matrix = self.Fisher_matrix - np.outer(self.av_grad, self.av_grad)
        #regularize 
        self.Fisher_matrix = self.Fisher_matrix + self.l_reg * np.eye(self.num_params)

    def reset(self):
        self.__init__(self.num_params, self.l_reg)

    def apply(self, grad):
        return np.matmul(np.linalg.inv(self.Fisher_matrix), grad)