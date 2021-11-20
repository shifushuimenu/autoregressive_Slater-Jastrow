"""Routines for bitcoded fermionic occupation number states."""
# TODO: - Replace numpy routines by torch routines (The default argument is requires_grad=False.)
#         everywhere and not just in places where a TypeError shows up.

import torch 
import numpy as np
from utils import default_torch_device 

def bin2int(bin_array):
    """
        Accepts batches. 
        Converts torch.Tensor to numpy.array. 
        
        Example:
        --------
        >>> s = np.array([[1,0,0,1],[0,0,1,1]])
        >>> bin2int(s)
        tensor([9, 3])
    """
    bin_array = np.array(bin_array, dtype=np.long)
    int_ = np.zeros(bin_array.shape[0:-1], dtype=np.long)
    for i in range(bin_array.shape[-1]):
        int_[...] = ( np.left_shift(int_[...], 1) ) | bin_array[..., i]
    return torch.as_tensor(int_, device=default_torch_device)


def int2bin(I, ns):
    """
        Accepts batches (arbitrary number of leading dimensions)
        
        Parameters:
        -----------
        ns: int 
            number of lattice sites 
        I: int or array of ints 
            Integer representing an occupation number state 
            
        Returns:
        --------
        bin_array: array of 0s and 1s 
            Occupation number state corresponding to I
            
        Example:
        --------
        >>> ns=4; I=[[10, 3],[3, 10]]
        >>> int2bin(I, ns)
        array([[[1, 0, 1, 0],
                [0, 0, 1, 1]],
        <BLANKLINE>
               [[0, 0, 1, 1],
                [1, 0, 1, 0]]])

    """
    # Hack. Is this fast enough ? 
    I = np.array(I, dtype=np.long)
    bin_array = np.zeros(I.shape + (ns,), dtype=np.long)
    scan = np.ones_like(I)
    for i in range(ns):
        bin_array[..., ns-i-1] = np.bitwise_and(I[...], np.left_shift(scan[...], i)) // np.left_shift(scan[...], i)
    return bin_array


def int2pos(I, ns):
    """
        Transform from integer `I` encoding an occupation number state to 
        an array containing the particle positions in the occupation number state. 
        This array is used for indexing into a matrix representing a Slater determinant.
        
        Parameters:
        -----------
            I: batch (!) of integers 
            ns: number of sites 
            
        Returns:
        --------
        
        Example:
        --------
        >>> I = np.array([[3,12,5],[5,6,6]], dtype=np.long)
        >>> int2pos(I, 5)
        array([[[3, 4],
                [1, 2],
                [2, 4]],
        <BLANKLINE>
               [[2, 4],
                [2, 3],
                [2, 3]]])


    """
    I = np.array(I, dtype=np.long)
    bin_array = int2bin(I, ns)
    return bin2pos(bin_array)
     

def bin2pos(bin_array):
    """
        Convert binary array of occupation numbers into array of 
        particle positions. Particle positions are counted from the 
        right.
        
        Example: (broadcast over arbitrary large number of leading dimensions)
        --------
        >>> bin_array = [[[0,1,1,0], [1,1,0,0], [0,1,0,1]], [[0,1,1,0], [1,1,0,0], [0,1,0,1]]]
        >>> bin2pos(bin_array)
        array([[[1, 2],
                [0, 1],
                [1, 3]],
        <BLANKLINE>
               [[1, 2],
                [0, 1],
                [1, 3]]])


    """    
    bin_array = np.array(bin_array, dtype=np.long)
    B0 = bin_array.reshape(-1, bin_array.shape[-1])
    # All occupation number states must be in the same particle number sector.
    Np = np.count_nonzero(B0[0,:])
    assert np.all(np.count_nonzero(B0, axis=1) == Np)    
    dummy = np.vstack([i_.nonzero()[0] for i_ in B0])   # i_[::-1] => The bit order is inversed because particle positions are counted from the right.
    pos_array = np.reshape(a=dummy, newshape=bin_array.shape[:-1] + (Np,))
    
    return pos_array   
    
def _test():
    import doctest 
    doctest.testmod(verbose=True)

if __name__ == "__main__":
    _test()
