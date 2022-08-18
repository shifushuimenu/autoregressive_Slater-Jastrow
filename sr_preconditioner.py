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
from sr_lazy import SR_Preconditioner_base
import torch 
import numpy as np
from dataclasses import dataclass

__all__ = ["SR_Preconditioner"]

class SR_Preconditioner(SR_Preconditioner_base):
    """This subclass provides the interface to gradients computed in torch."""
    def __init__(self, num_params, num_samples, eps1=0.0, eps2=0.0, lazy=True, dtype=np.float64):
        super(SR_Preconditioner, self).__init__(num_params, num_samples, eps1=eps2, eps2=eps2, lazy=lazy, dtype=dtype)

    def _flatten(self, grad):
        """The gradient (and also O_k)  is a list of torch tensor matrices and vectors (same as module.parameters()).
        Flatten it into a single continuous vector. Input `grad`is a numpy array. """
        grad_flat = np.hstack([p.flatten() for p in grad])
        return grad_flat 

    def _unflatten(self, grad_flat, grad):
        """Use information in `grad` to reassemble flattened gradient into blocks of parameters."""
        offset = 0
        grad_new = []
        for ps in [list(p.size()) for p in grad]:
            param_block = grad_flat[offset:offset+np.prod(ps)].reshape(ps)
            grad_new.append(torch.tensor(param_block))
            offset += np.prod(ps)
        return grad_new  
        
    def accumulate(self, O_k):
       super(SR_Preconditioner, self).accumulate(self._flatten(O_k))

    def apply_Sinv(self, g, tol=1e-8):
        g_flat = self._flatten(g)
        x = super(SR_Preconditioner, self).apply_Sinv(g_flat, tol=tol)
        # we don't need the Fisher information matrix any longer
        self.reset()
        return self._unflatten(x, g)

@dataclass
class Identity_Preconditioner(SR_Preconditioner_base):
    """Dummy class for performing SGD without SR"""
    x : int = 0

    def accumulate(self, O_k):
        pass 

    def apply_Sinv(self, g, tol=1e-5):
        return g
