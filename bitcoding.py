"""
Routines for bitcoded fermionic occupation number states.

dtype='object'  ensures that integers can be as large as memory allows.
"""
# TODO: - Replace numpy routines by torch routines (The default argument is requires_grad=False.)
#         everywhere and not just in places where a TypeError shows up.
#       - Bit are ordered frmo left to right. Oringinally, they were ordered from right to left. 
#         The change has introduced some hacks, which should be eliminated in favour of a cleaner solution.
#
#       - Use a class for basis states where both the numpy binary array and the integer value are stored. 
#         This avoids repeated conversions.  
import numpy as np
from utils import default_torch_device 


def fermion_parity( n, state_idx, i, j ):
    """
    Starting from the occupation number state encoded by the integer 
    `state_idx`, let a particle hop from position `i` to position `j`
    (or the backward process, i<j), which may result in a new state. If the new 
    state does not vanish, fermion_parity() returns its sign. 

    So this functions counts the number of ones in the bit representation of 
    integer `state_idx` between sites i and j, i.e. in the closed interval [i+1, j-1]. 
    
    Parameters:
    -----------
        n: number of sites   
        state_idx: int or 1d array_like of ints 
            bitcoded occupation number state 
        i, j: ints
            0 <= i < j < n. The particle is assumed to hop from i to j or from j to i,
            irrespective of whether such a hopping process is possible for the given 
            occupation number states. 
        
    Returns:
    --------
        parity: \in [+1, -1]
        
    Example:
    --------
    >>> l = [7, 10, 0] # accepts list 
    >>> fermion_parity( 4, l, 0, 3 )
    array([ 1, -1,  1])
    >>> a = np.array([6, 10, 0]) # accepts numpy array 
    >>> fermion_parity( 4, a, 0, 3 )
    array([ 1, -1,  1])
        
    """
    state_idx = np.array(state_idx, dtype='object')
    #assert(np.all(state_idx < pow(2,n)))
    #assert(0 <= i < j < n)
    # count number of particles between site i and j 
    mask = np.zeros((state_idx.shape + (n,)), dtype='object')
    mask[..., slice(i+1, j)] = 1
    mask = bin2int(mask)
    num_exchanges = np.array(
        [bin(np.bitwise_and(mask[batch_idx], state_idx[batch_idx])).count('1') for batch_idx in range(mask.shape[0])]
        )      
    parity = np.where(num_exchanges%2==0, +1, -1)
    
    return parity         


def fermion_parity2(n, state_idx, i, j):
    """
    FASTER, NO BATCH INPUT ALLOWED 

    Starting from the occupation number state encoded by the integer 
    `state_idx`, let a particle hop from position `i` to position `j`
    (or the backward process, i<j), which may result in a new state. If the new 
    state does not vanish, fermion_parity() returns its sign. 

    So this functions counts the number of ones in the bit representation of 
    integer `state_idx` between sites i and j, i.e. in the closed interval [i+1, j-1]. 

    s = '0b11001' -> s[2:] = '11001'  (remove the leading characters 0b)
    s[2:].rjust(6, '0') = '011001'    (add leading zeros by right-justifying)
    s[2:].rjust(6, '0')[::-1] = '100110'  (invert order because in our convention we count sites from the right)

    
    Parameters:
    -----------
        n: number of sites   
        state_idx: int or 1d array_like of ints 
            bitcoded occupation number state 
        i, j: ints
            0 <= i < j < n. The particle is assumed to hop from i to j or from j to i,
            irrespective of whether such a hopping process is possible for the given 
            occupation number states. 
        
    Returns:
    --------
        parity: \in [+1, -1]

    Example:
    --------
    >>> fermion_parity2(4, 6, 0, 3)
    1
    >>> fermion_parity2(4, 10, 0, 3)
    -1
    """
    #assert 0 <= i < j < n
    num_exchanges = bin(state_idx)[2:].rjust(n, '0')[::-1].count('1', i+1, j)
    return - 2 * (num_exchanges%2) + 1


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


def bin2int_nobatch(bin_array):
    """
    """
    assert len(bin_array.shape) == 1 # no batch dimension 
    bin_array = bin_array[::-1]
    int_ = 0
    for i in range(bin_array.shape[-1]):
        int_ = (int_ << 1) | int(bin_array[i])
    return int_


def bin2int_nobatch_v2(bin_array):
    """
    """
    bin_array_s = list(map(str, bin_array[::-1]))
    S = "0b" + "".join([b for b in bin_array_s])
    return int(S, base=0)


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
                [0, 1, 0, 1]]])

    """
    # Hack. Is this fast enough ? 
    I = np.array(I, dtype='object')
    bin_array = np.zeros(I.shape + (ns,), dtype=int)
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
