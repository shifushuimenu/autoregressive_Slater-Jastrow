import numpy as np
import torch 

from utils import default_dtype_torch

from profilehooks import profile

#@profile
def block_update_inverse(Ainv, B, C, D):
    """
        Inputs are torch tensors. 

        Use the update formula for the inverse of a block matrix
            M = [[ A , B ]
                 [ C ,  D]]
        taking into account that Ainv is already known. 
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
    assert torch.isclose(C, B.transpose(-1,-2)).all()

    AinvB = torch.matmul(Ainv, B)
    S = D - torch.matmul(C, AinvB) # a scalar 
    Sinv = 1.0/S.item()
    
    Ablock = Ainv + torch.outer(AinvB[:,0], Sinv * AinvB[:,0])
    Bblock = - AinvB * Sinv
    #Cblock = Bblock.transpose(-1,-2)
    Dblock = Sinv 

    output = torch.empty(n+1, n+1, dtype=default_dtype_torch)

    output[0:n, 0:n] = Ablock
    output[0:n, n] = Bblock[:,0]
    output[n, 0:n] = Bblock[:,0]
    output[n, n] = Dblock 

    return output 