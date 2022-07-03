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
import torch

# TODO: introduce regularization parameter before inverting S

class SR_Preconditioner(object):
    def  __init__(self, num_params, l_reg=1e-8):
        self.num_params = num_params 
        assert l_reg > 0 
        self.l_reg = l_reg
        self.Fisher_matrix = np.zeros((num_params, num_params), dtype=np.float64)
        self.av_grad = np.zeros((num_params), dtype=np.float64)
        self.counter = 0

    def acc_Fisher_infomatrix(self, grad):
        """$grad_ k= \frac{\partial_k |\psi\rangle}{|\psi\rangle}$"""
        self.counter += 1 
        grad_flat = self._flatten(grad)
        self.Fisher_matrix += np.outer(grad_flat, grad_flat)
        self.av_grad += grad_flat

    def av_Fisher_infomatrix(self):
        self.Fisher_matrix /= self.counter 
        self.av_grad /= self.counter 

        self.Fisher_matrix = self.Fisher_matrix - np.outer(self.av_grad, self.av_grad)
        # regularize 
        self.Fisher_matrix = self.Fisher_matrix + self.l_reg * np.eye(self.num_params)

    def reset(self):
        self.__init__(self.num_params, self.l_reg)

    def apply(self, grad):
        grad_flat = self._flatten(grad)
        # The Fisher information matrix, as a covariance matrix, must be positive definite. 
        if __debug__:
            e = np.linalg.eigvals(self.Fisher_matrix)
            assert( np.all(e >= 0) )
        Sinvg = np.matmul(np.linalg.inv(self.Fisher_matrix), grad_flat)
        return self._unflatten(grad, Sinvg)

    def _flatten(self, grad):
        """The gradient is a list of torch tensor matrices and vectors (same as module.parameters()).
        Flatten it into a single continuous vector."""
        grad_flat = np.hstack([p.flatten() for p in grad])
        return grad_flat 

    def _unflatten(self, grad, grad_flat):
        """Use information in `grad` to reassemble flattened gradient into blocks of parameters."""
        offset = 0
        grad_new = []
        for ps in [list(p.size()) for p in grad]:
            param_block = grad_flat[offset:offset+np.prod(ps)].reshape(ps)
            grad_new.append(torch.tensor(param_block))
            offset += np.prod(ps)
        return grad_new