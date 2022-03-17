"""
Routines for bitcoded fermionic occupation number states.

dtype='object'  ensures that integers can be as large as memory allows.
"""
# TODO: - Replace numpy routines by torch routines (The default argument is requires_grad=False.)
#         everywhere and not just in places where a TypeError shows up.
#       - Bit are ordered frmo left to right. Oringinally, they were ordered from right to left. 
#         The change has introduced some hacks, which should be eliminated in favour of a cleaner solution.

import torch 
import numpy as np
from utils import default_torch_device 

def bin2int(bin_array):
    """
        First dimension is batch dimension (obligatory).
        
        Example:
        --------
        >>> s = np.array([[1,0,0,1],[0,0,1,1]])
        >>> bin2int(s)
        array([9, 12], dtype=object)
    """
    bin_array = np.array(bin_array[...,::-1], dtype='object') #)   # hack
    int_ = np.zeros(bin_array.shape[0:-1], dtype='object') #np.compat.long)
    for i in range(bin_array.shape[-1]):
        int_[...] = ( np.left_shift(int_[...], 1) ) | bin_array[..., i]
    return int_
    #return torch.as_tensor(int_, device=default_torch_device)

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
        array([[[0, 1, 0, 1],
                [1, 1, 0, 0]],
        <BLANKLINE>
               [[1, 1, 0, 0],
                [0, 1, 0, 1]]], dtype=object)

    """
    # Hack. Is this fast enough ? 
    I = np.array(I, dtype='object')
    bin_array = np.zeros(I.shape + (ns,), dtype='object')
    scan = np.ones_like(I)
    for i in range(ns):
        bin_array[..., ns-i-1] = np.bitwise_and(I[...], np.left_shift(scan[...], i)) // np.left_shift(scan[...], i)
    bin_array = bin_array[..., ::-1] # hack
    return np.array(bin_array)  # np.array() is necessary to avoid np-array with negative strides  


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
        >>> I = np.array([[3,12,5],[5,6,6]], dtype='object')
        >>> int2pos(I, 5)
        array([[[0, 1],
                [2, 3],
                [0, 2]],
        <BLANKLINE>
               [[0, 2],
                [1, 2],
                [1, 2]]])


    """
    I = np.array(I, dtype='object')
    bin_array = int2bin(I, ns)
    return bin2pos(bin_array)
     

def bin2pos(bin_array):
    """
        Convert binary array of occupation numbers into array of 
        particle positions. Particle positions are counted from the 
        left.
        
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
    bin_array = np.array(bin_array, dtype='object')
    B0 = bin_array.reshape(-1, bin_array.shape[-1])
    # All occupation number states must be in the same particle number sector.
    Np = np.count_nonzero(B0[0,:])
    assert np.all(np.count_nonzero(B0, axis=1) == Np)    
    dummy = np.vstack([i_.nonzero()[0] for i_ in B0])   
    pos_array = np.reshape(a=dummy, newshape=bin_array.shape[:-1] + (Np,))
    
    return pos_array   
    
def _test():
    import doctest 
    doctest.testmod(verbose=True)

if __name__ == "__main__":
    _test()
