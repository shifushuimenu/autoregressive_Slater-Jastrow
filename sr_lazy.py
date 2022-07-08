from scipy.sparse.linalg import LinearOperator, cg

import numpy as np
import torch 

__all__ = ["SR_Preconditioner_base"]

class SR_Preconditioner_base(object):
    """
    $O_k(x) = \frac{\partial log \psi(x)}{\partial \theta_k}$ is the logarithmic derivative 
    of the wavefunction with respect to parameter $\theta_k$ evaluated at sample $x$. 
    """
    def __init__(self, num_params, num_samples, diag_shift=0.0, lazy=True, dtype=np.float64):
        self.num_params = num_params 
        self.num_samples = num_samples 
        self.diag_shift= diag_shift
        self.dtype = dtype
        self.O_ks = np.zeros((self.num_params, self.num_samples))
        self.scale = np.zeros((self.num_params,))
        self.isample = 0

    def accumulate(self, O_k):
        O_k = np.asarray(O_k, dtype=self.dtype)
        assert O_k.shape[0] == self.num_params and len(O_k.shape)==1
        self.O_ks[:, self.isample] = O_k[:]
        self.isample += 1
        self.ctr = False

    def center(self):
        """calculate (O_k - <O_k>)"""
        assert self.isample == self.num_samples
        self.O_ks = self.O_ks - np.sum(self.O_ks, axis=-1)[:,None] / self.num_samples
        # diagonal matrix elements of centered S-matrix 
        self.scale = np.sum(np.array([self.O_ks[:, ii] * self.O_ks[:, ii] for ii in np.arange(self.num_samples)]), axis=0)
        self.ctr = True

    def matvec(self, v):
        """lazy matrix-vector product"""
        if not self.ctr:
            self.center()
        v = np.asarray(v, dtype=self.dtype)
        mv1 = np.matmul(self.O_ks.T, v)
        mv2 = np.matmul(self.O_ks, mv1)
        return mv2 + self.diag_shift*v

    def _to_dense(self):
        """convert the lazy matrix representation to a dense matrix representation"""
        if not self.ctr:
            self.center()
        S = np.zeros((self.num_params, self.num_params))
        for ii in np.arange(self.num_samples):
            S += np.outer(self.O_ks[:,ii], self.O_ks[:,ii])
        return S

    def rescale_diag(self, v):
        """use as Jacobi preconditioner"""
        assert self.ctr
        return v / self.scale

    def reset(self):
        self.__init__(self.num_params, self.num_samples, self.diag_shift, lazy=True, dtype=self.dtype)     