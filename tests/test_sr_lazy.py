import sys, os

testdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, testdir+'/..')


import unittest 
import numpy as np

from sr_lazy import SR_Preconditioner_base

class TestSR(unittest.TestCase):

    def test_lazy_matrix_representation(self, tol=1e-8):
        # linear operator representing the overlap matrix 
        npara = 100; ns = 1000
        SR = SR_Preconditioner_base(num_params=npara, num_samples=ns, eps1=0.0, eps2=0.0, dtype=np.float64)
        O_ks = np.random.randn(npara, ns)
        for ii in range(ns):
            SR.accumulate(O_ks[:, ii])
        SR._center()

        b = np.random.randn(npara)
 
        x1 = SR.apply_Sinv(b, tol=tol)

        Smat = SR._to_dense()
        assert np.all(np.isclose(SR.diagme, np.diag(Smat), atol=tol)) # SR.scale -> SR.diagme

        x2 = np.linalg.inv(Smat) @ b
        assert np.all(np.isclose(x1, x2, atol=tol))        

if __name__ == "__main__":
    unittest.main()
