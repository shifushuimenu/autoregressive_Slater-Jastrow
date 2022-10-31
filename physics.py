"""Routines for fermionic lattice Hamiltonians: lattice and kinetic term"""
import torch
import numpy as np

from bitcoding import *
from utils import default_torch_device
# from profilehooks import profile

__all__ = ["fermion_parity", "kinetic_term", "Lattice_rectangular"]


class Lattice1d(object):
    """1d lattice with pbc"""
    def __init__(self, ns=4):
        self.ns = ns 
        self.nx = ns
        self.ny = 1
        self.coord = 2 
        self.neigh = np.zeros((self.ns, self.coord), dtype='object')
        # left neighbours 
        self.neigh[0, 0] = self.ns-1
        self.neigh[1:, 0] = range(0, self.ns-1)
        # right neighbours 
        self.neigh[:-1,1] = range(1, self.ns)
        self.neigh[ns-1, 1] = 0


class Lattice_rectangular(object):
    def __init__(self, nx=4, ny=4):
        self.nx = nx 
        self.ny = ny 
        self.ns = self.nx * self.ny
        self.coord = 4

        rectl = np.arange(self.ns).reshape(self.nx, self.ny) 
        up    = np.roll(rectl, 1, axis=0).flatten()
        right = np.roll(rectl, -1, axis=1).flatten()
        down  = np.roll(rectl, -1, axis=0).flatten()
        left  = np.roll(rectl, 1, axis=1).flatten()        

        self.neigh = np.vstack((up, right, down, left)).transpose().astype('object') # idxs: sitenr, direction (up=0, right=1, down=2, left=3)        

        # hopping bonds 
        self.bonds = []            
        for i in range(self.ns):
            for d in range((self.coord//2)):
                j = self.neigh[i, d]
                self.bonds.append((i,j))
        self.num_bonds = len(self.bonds)
        assert self.num_bonds == 2*self.ns



def fermion_parity(n, state_idx, i, j):
    """
    NO BATCH INPUT ALLOWED.

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
    >>> fermion_parity(4, 6, 0, 3)
    1
    >>> fermion_parity(4, 10, 0, 3)
    -1
    """
    #assert 0 <= i < j < n
    num_exchanges = bin(state_idx)[2:].rjust(n, '0')[::-1].count('1', i+1, j)
    return (-2) * (num_exchanges%2) + 1


#@profile
def kinetic_term( I, lattice, t_hop=1.0 ):
    """
        NO BATCH DIMENSION. 

        Parameters:
        -----------
            I: Bitcoded integer representing occupation numbers 
               of spinless fermions.
            lattice: Lattice object 
                Provides nearest neighbour matrix which defines the possible 
                hopping terms. 
            t_hop: ( optional )
                hopping parameter 
            
        Returns:
        --------
            hop_from_to: list of pairs [(i1_initial, i1_final), (i2_initial, i2_final), ...]
                where state I1_prime is obtained from I by a particle hopping from i1_initial to i1_final, 
                state I2_prime is obtained from I by hopping from i2_initial to i2_final etc. 
            I_prime: list of ints of length `num_bonds`
                List of states connected to I by the application of the kinetic operator K_kin.
                `num_bonds` is the number of distinct hopping terms in the kinetic
                operator. If a given hopping term annihilates state |I>, the "connecting state" is still recorded, 
                however with matrix element zero. This is to ensure that, given a batch of samples, 
                            
            matrix_elem: list of floats             
                <I| K_kin |I_prime> for all possible I_prime given the lattice structure. 

        Example:
        --------
        >>> rctl = Lattice_rectangular(4,4)
        >>> hop_from_to, I_prime, matrix_elem = kinetic_term(0+2**3+2**5+2**7+2**10+2**12, rctl)
        >>> hop_from_to
        [(12, 0), (3, 2), (3, 15), (3, 0), (5, 4), (5, 1), (5, 6), (7, 6), (7, 4), (5, 9), (10, 9), (10, 6), (10, 11), (7, 11), (12, 8), (12, 13), (10, 14), (12, 15)]
        >>> I_prime[0:8]
        array([1193, 5284, 38048, 5281, 5272, 5258, 5320, 5224], dtype=object)
        >>> matrix_elem[0:8]
        array([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0], dtype=object)
    """
    #I = np.asarray(I, dtype='object')
    assert type(I) == int 
    neigh = lattice.neigh
    ns = lattice.ns
    coord = lattice.coord

    # preallocate
    I_prime = np.empty((lattice.num_bonds, ), dtype='object')
    matrix_elem = np.empty_like(I_prime)
    rs_pos = [] # particle hopping from position r to position s

    count = 0
    for bond in lattice.bonds:
        (i,j) = bond
        pow2i = 1 << i; pow2j = 1 << j # 2**i and 2**j
        M = pow2i + pow2j
        K = M & I # bitwise AND
        L = K ^ M # bitwise XOR
        STATE_EXISTS = ((K != 0) & (L != 0) & (L != K))
        if STATE_EXISTS:
            I_prime[count] = I - K + L
            ii = min(i,j)
            jj = max(i,j)
            matrix_elem[count] = -t_hop * fermion_parity(ns, I, ii, jj)

            if I & pow2i == pow2i and I ^ I+pow2j == pow2j:
                r = i; s = j
            elif I & pow2j == pow2j and I ^ I+pow2i == pow2i:
                r = j; s = i 
            else:
                r = -1; s = -1
            rs_pos.append((r,s))
            count += 1

    I_prime = I_prime[0:count]
    matrix_elem = matrix_elem[0:count]

    # make sure there are no duplicates in the hopping bonds (this happens for 2x2 lattice )
    rs_pos_unique = []
    idx_unique = []
    for ii, item in enumerate(rs_pos):
        if item not in rs_pos_unique:
            rs_pos_unique.append(item)
            idx_unique.append(ii)
    rs_pos = rs_pos_unique 
    I_prime = I_prime[idx_unique]
    matrix_elem = matrix_elem[idx_unique]

    return ( rs_pos, I_prime, matrix_elem )


def _test():
    import doctest 
    doctest.testmod(verbose=True)


if __name__ == "__main__":
    import doctest 
    _test()
