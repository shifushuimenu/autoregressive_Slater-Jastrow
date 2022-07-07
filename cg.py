"""Conjugate gradient algorithm for solving Ax=b for symmetric positive definite matrix A."""
import numpy as np

def conjugate_gradient(A, b, x0, eps, kmax):
    """A is a function realizing the matrix vector product"""
    rho = np.zeros(kmax+1, dtype=np.float64)
    beta = np.zeros(kmax+1, dtype=np.float64)
    alpha = np.zeros(kmax+1, dtype=np.float64)
    epsnormb = eps*np.sqrt(np.dot(b, b))

    k=0
    x = x0
    r = b - A(x)
    rho[0] = np.dot(r, r)
    while ( np.sqrt(rho[k]) > epsnormb and k < kmax ):
        k += 1 
        if k == 1:
            p = r
        else:
            beta[k] = rho[k-1] / rho[k-2]
            p = r + beta[k] * p
        w = A(p)
        alpha[k] = rho[k-1] / np.dot(p, w)
        x = x + alpha[k] * p
        r = r - alpha[k] * w
        rho[k] = np.dot(r, r)

    return x, rho[k], k