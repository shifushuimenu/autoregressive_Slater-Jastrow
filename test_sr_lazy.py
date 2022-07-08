import unittest 
from sr_lazy import SR_Preconditioner

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg 


class TestSR(unittest.TestCase):

    def test_lazy_matrix_representation(self, tol=1e-8):
        # linear operator representing the overlap matrix 
        npara = 50; ns = 100
        SR = SR_Preconditioner(num_params=npara, num_samples=ns, diag_shift=0.0, dtype=np.float64)
        O_ks = np.random.randn(npara, ns)
        for ii in range(ns):
            SR.accumulate(O_ks[:, ii])
        SR.center()
        S = LinearOperator(shape=(npara, npara), matvec=SR.matvec)
        # Jacobi preconditioner
        M = LinearOperator(shape=(npara, npara), matvec=SR.rescale_diag)
        b = np.random.randn(npara)

        x1, info = cg(S, b, tol=tol, atol=0, M=M)
        assert info == 0
 
        Smat = SR._to_dense()
        assert np.all(np.isclose(SR.scale, np.diag(Smat), atol=tol))

        x2 = np.linalg.inv(Smat) @ b
        assert np.all(np.isclose(x1, x2, atol=tol))        

if __name__ == "__main__":
    unittest.main()