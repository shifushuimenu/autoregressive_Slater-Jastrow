import unittest 
import numpy as np

from cg import conjugate_gradient 

class Test1(unittest.TestCase):

    def solve_Axb(self):
        n=10
        L = np.random.randn(n,n)
        A = L.T @ L # positive definite matrix 
        assert np.all(np.linalg.eigvals(A) > 0)
        b = np.random.randn(n)
        x0 = np.zeros(10)

        # solve A*x = b
        x, res, k = conjugate_gradient(lambda x: A@x, b, x0, eps=1e-8, kmax=10)

        self.assertTrue(np.isclose(a=np.linalg.norm(x - np.linalg.inv(A) @ b, ord=2), b=0.0, atol=1e-4))

if __name__ == "__main__":
    unittest.main()
