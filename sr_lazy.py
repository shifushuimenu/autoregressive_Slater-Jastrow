import numpy as np
from scipy.sparse.linalg import LinearOperator, cg 

__all__ = ["SR_Preconditioner_base"]

class SR_Preconditioner_base(object):
    """
    $O_k(x) = \frac{\partial log \psi(x)}{\partial \theta_k}$ is the logarithmic derivative 
    of the wavefunction with respect to parameter $\theta_k$ evaluated at sample $x$. 

    Parameters:
    -----------
  
    eps1, eps2 : float 
        Regularization of the S-matrix:
          S_{m,m} -> S_{m,m} * (1+eps1) + eps2

        Diagonal elements of the matrix S are rescaled by (1+eps1) and shifted by eps2.
    """
    def __init__(self, num_params, num_samples, eps1=0.0, eps2=0.0, lazy=True, dtype=np.float64):
        self.num_params = num_params 
        self.num_samples = num_samples 
        self.eps1 = eps1
        self.eps2 = eps2
        self.dtype = dtype
        self.O_ks = np.zeros((self.num_params, self.num_samples))
        self.diagme = np.zeros((self.num_params,))
        self.max_diagme = 0.0
        self.isample = 0

    def accumulate(self, O_k):
        O_k = np.asarray(O_k, dtype=self.dtype)
        assert O_k.shape[0] == self.num_params and len(O_k.shape)==1
        self.O_ks[:, self.isample] = O_k[:]
        self.isample += 1
        self.ctr = False

    def _center(self):
        """calculate (O_k - <O_k>)"""
        assert self.isample == self.num_samples
        self.O_ks = self.O_ks - np.sum(self.O_ks, axis=-1)[:,None] / self.num_samples
        # diagonal matrix elements of centered S-matrix 
        self.diagme = np.sum(np.array([self.O_ks[:, ii] * self.O_ks[:, ii] for ii in np.arange(self.num_samples)]), axis=0)
        self.max_diagme = np.max(self.diagme)
        self.ctr = True

    def _matvec(self, v):
        """lazy matrix-vector product"""
        if not self.ctr:
            self.center()
        v = np.asarray(v, dtype=self.dtype)
        mv1 = np.matmul(self.O_ks.T, v)
        mv2 = np.matmul(self.O_ks, mv1)
        return mv2 + (self.eps2 * self.max_diagme + self.eps1 * self.diagme[:])*v

    def _rescale_diag(self, v):
        """use as Jacobi preconditioner"""
        assert self.ctr
        v_rescaled = np.where(self.diagme > 1e-8, v / self.diagme, v)
        return v_rescaled

    def _to_dense(self):
        """convert the lazy matrix representation to a dense matrix representation"""
        if not self.ctr:
            self.center()
        S = np.zeros((self.num_params, self.num_params))
        for ii in np.arange(self.num_samples):
            S += np.outer(self.O_ks[:,ii], self.O_ks[:,ii])
        return S        

    def reset(self):
        self.__init__(self.num_params, self.num_samples, eps1=self.eps1, eps2=self.eps2, lazy=True, dtype=np.float64)     

    def apply_Sinv(self, v, tol=1e-8, Jacobi_precond=False):
        """ x = S^{-1}@v using conjugate gradient with lazy matrix evaluation"""
        if not self.ctr:
            self._center()       

        S = LinearOperator(shape=(self.num_params, self.num_params), matvec=self._matvec)
        # Jacobi preconditioner (rescale by diagonal matrix elements)
        if Jacobi_precond:
            M = LinearOperator(shape=(self.num_params, self.num_params), matvec=self._rescale_diag)
        else:
            M=None
        # preconditioned conjugate gradient method 
        x, info = cg(S, b=v, tol=tol, atol=0, M=M)
        assert info == 0

        return x

