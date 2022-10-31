import numpy as np

from utils import default_dtype_torch

#from profilehooks import profile
from my_linalg_numpy import my_det 

#@profile
def block_update_inverse(Ainv, B, C, D):
    """
    Inputs are torch tensors. 

    Use the update formula for the inverse of a block matrix
        M = [[ A , B ]
                [ C ,  D]]
    taking into account that A^{-1} = Ainv is already known. 
    A is an nxn matrix and B and C are assumed to be nx1 and 1xn matrices,
    respectively. D is assumed to be a 1x1 matrix.
    Furthermore, C = B.transpose().

    Minv =  [[ Ablock , Bblock ]
                [ Cblock , Dblock ]]
    """
    assert Ainv.shape[0] == Ainv.shape[1]
    n = Ainv.shape[0]
    assert B.shape[0] == n and B.shape[1] == 1
    assert C.shape[0] == 1 and C.shape[1] == n
    assert D.shape[0] == D.shape[1] == 1 
    assert np.isclose(C, B.transpose(-1,-2)).all()

    AinvB = np.matmul(Ainv, B)
    S = D - np.matmul(C, AinvB) # a scalar 
    Sinv = 1.0/S
    
    Ablock = Ainv + np.outer(AinvB[:,0], Sinv[0,0] * AinvB[:,0])
    Bblock = - AinvB * Sinv
    #Cblock = Bblock.transpose(-1,-2)
    Dblock = Sinv 

    output = np.empty((n+1, n+1))

    output[0:n, 0:n] = Ablock
    output[0:n, n] = Bblock[:,0]
    output[n, 0:n] = Bblock[:,0]
    output[n, n] = Dblock 

    return output 


def block_update_inverse2(Ainv, B, C, D):
    """
    The same as block_update_inverse(...) except that the
    matrices B, C, and D are general nxm matrices rather than vectors.
    """
    assert Ainv.shape[0] == Ainv.shape[1]
    n = Ainv.shape[0]
    assert B.shape[0] == n
    m = B.shape[1]
    assert C.shape[0] == m and C.shape[1] == n
    assert D.shape[0] == D.shape[1] == m
    assert np.isclose(C, B.transpose(-1,-2)).all()

    AinvB = np.matmul(Ainv, B)
    S = D - np.matmul(C, AinvB)  # Schur complement of A in M
    Sinv = np.linalg.inv(S)

    Ablock = Ainv + np.matmul(np.matmul(AinvB, Sinv), AinvB.transpose())
    Bblock = - np.matmul(AinvB, Sinv)
    Cblock = Bblock.transpose(-1,-2)
    Dblock = Sinv 

    output = np.empty((n+m, n+m))

    output[0:n, 0:n] = Ablock
    output[0:n, n:n+m] = Bblock
    output[n:n+m, 0:n] = Cblock
    output[n:n+m, n:n+m] = Dblock 

    return output     


def block_update_det_correction(Ainv, B, C, D):
    """
    Let  M = [[ A , B ]
                [ C ,  D]]
    and A^{-1} = Ainv is already known. Then: 
        det(M) = det(A) * det(D - C A^{-1} B)
    This function returns the correction factor 
        corr = det(D - C A^{-1} B)
    """
    assert Ainv.shape[0] == Ainv.shape[1]
    n = Ainv.shape[0]
    assert B.shape[0] == n and B.shape[1] == 1
    assert C.shape[0] == 1 and C.shape[1] == n
    assert D.shape[0] == D.shape[1] == 1 

    AinvB = np.matmul(Ainv, B)
    corr = D - np.matmul(C, AinvB)
    return corr


def block_update_det_correction2(Ainv, B, C, D):
    """
    The same as above except that B,C, and D are matrices rather than vectors. 
    """
    assert Ainv.shape[0] == Ainv.shape[1]
    n = Ainv.shape[0]
    assert B.shape[0] == n
    m = B.shape[1]
    assert C.shape[0] == m and C.shape[1] == n
    assert D.shape[0] == D.shape[1] == m
    assert np.isclose(C, B.transpose(-1,-2)).all()

    AinvB = np.matmul(Ainv, B)
    S = D - np.matmul(C, AinvB)
    # For small matrices the determinant operation should be hand-coded 
    # for speed-up. 
    #return np.linalg.det(S)  # determinant of the Schur complement
    return my_det(S, m)  # determinant of the Schur complement
