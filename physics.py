"""Routines for fermionic lattice Hamiltonians"""
import torch
import numpy as np

from bitcoding import *
from utils import default_torch_device

#__all__ = ["local_estimator", "ham"]

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
                0 <= i < j < n. The particle is assumed to hop from i to j or form j to i,
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
        >>> t = torch.Tensor([6, 10, 0]) # accepts torch tensor
        >>> fermion_parity( 4, t, 0, 3 )
        array([ 1, -1,  1])
        
    """
    state_idx = np.array(state_idx, dtype=np.int64)
    assert(np.all(state_idx < np.power(2,n)))
    assert(0 <= i < j < n)
    # count number of particles between site i and j 
    mask = np.zeros((state_idx.shape + (n,)), dtype=np.int64)
    mask[..., slice(i+1, j)] = 1
    mask = bin2int(mask)
    num_exchanges = np.array(
        [bin(np.bitwise_and(mask[batch_idx], state_idx[batch_idx])).count('1') for batch_idx in range(mask.shape[0])]
        )      
    parity = np.where(num_exchanges%2==0, +1, -1)
    
    return parity         


###############################
# Just for testing purposes
class Lattice1d(object):
    def __init__(self, ns=4):
        self.ns = ns 
        self.coord = 2 
        self.neigh = np.zeros((self.ns, self.coord), dtype=np.int64)
        # left neighbours 
        self.neigh[0, 0] = self.ns-1
        self.neigh[1:, 0] = range(0, self.ns-1)
        # right neighbours 
        self.neigh[:-1,1] = range(1, self.ns)
        self.neigh[ns-1, 1] = 0
###############################

        
def kinetic_matrix_elem( I, lattice ):
    """
    Parameters:
    -----------
        I: batch of integers
           Bitcoded integer representing occupation numbers 
           of spinless fermions.
        lattice: Lattice object 
           Provides nearest neighbour matrix which defines the possible 
           hopping terms. 
        
    Returns:
    --------
        I_prime: list of ints of length `max_num_connect`
           List of states connected to I by the application of the kinetic operator K_kin.
           `max_num_connect` is the number of distinct hopping terms in the kinetic
           operator. If a given hopping term annihilates state |I>, the "connecting state" is still recorded, 
           however with matrix element zero. This is to ensure that, given a batch of samples, 
           I_prime has the same shape for every sample, although different occupation number 
           states are connected to a different number of other states by action of the hopping operator.
                        
        matrix_elem: list of floats             
           <I| K_kin |I_prime> for all possible I_prime given the lattice structure. 

    Example:
    --------
    """
    I = np.array(I, dtype=np.int64)
    neigh = lattice.neigh
    ns = lattice.ns
    coord = neigh.shape[-1]
    max_num_connect = ns*(coord//2)
    
    I_prime = np.empty(I.shape + (max_num_connect,), dtype=np.int64)
    matrix_elem = np.empty_like(I_prime)
    count = -1
    for d in range((coord//2)): ####### Replace this error-prone hack by a sum over hopping-bonds. 
        for i in range(ns):     #######
            M = np.zeros_like(I, dtype=np.int64)
            count += 1 
            j = neigh[i,d]
            ii = min(i,j)
            jj = max(i,j)
            M[...] = 2**ii + 2**jj
            K = np.bitwise_and(M, I)
            L = np.bitwise_xor(K, M)
            STATE_EXISTS = ((K != 0) & (L != 0) & (L != K))
            I_prime[..., count] = np.where(STATE_EXISTS, I - K + L, False)
            matrix_elem[..., count] = np.where(STATE_EXISTS, fermion_parity(ns, I, ii, jj), 0)
    return (torch.as_tensor(I_prime, device=default_torch_device), torch.as_tensor(matrix_elem, device=default_torch_device))
               

def interaction_energy( sample, V=+1.0 ):
    ####### Works only for 1D.
    sample = torch.as_tensor(sample)
    # diagonal matrix elements 
    # nearest-neighbour interactions
    interaction = sample[...,1:] * sample[...,:-1]
    interaction = interaction.sum(axis=-1)
    interaction += sample[...,0] * sample[...,-1]
    
    return V*interaction
    
    
def ham( sample ):
    """
        Hamiltonian for one-dimensional t-V model.

        Accepts batched input.

        Input:
        ------
        sample: binary array representing an occupation number state 
                First dimension is batch dimension. 

        Returns: 
        --------
        tuple: (connecting_states, matrix_elements)
        `connecting_states` is an array of integer-coded states that 
            are connected to the input state `sample` by the action of the 
            Hamiltonian. `connecting_states[..., 0]` is identical to `bin2int(sample[..., 0])`,
            i.e. `connecting_states` includes the input state. 
            First dimension is the batch dimension. 
        `matrix_elements` are ordered like the `connecting_states`.

    """
    sample = np.array(sample)

    num_particles = np.count_nonzero(sample[0,...])    
    # ========================
    # just for testing     
    V = 0.33
    t = 1.0
    ns = sample.shape[-1]
    lattice = Lattice1d(ns)
    # ========================    

    I = bin2int(sample)
    I_prime, matrix_elem = kinetic_matrix_elem( I, lattice )
    # Set states in `I_prime` which were annihilated by the hopping operator to a valid value (i.e.
    # in the correct particle number sector) so that downstream subroutines don't crash. 
    # This does not introduce errors, since the corresponding matrix elements are zero anyways
    # so that their multiplicative contribution is zero. 
    I_prime[I_prime == 0] = 2**num_particles - 1

    E_int = interaction_energy( sample )

    return ( 
        torch.cat((I[:,None], I_prime), dim=-1), # combine diagonal and off-diagonal matrix elements
        torch.cat((V*E_int[:,None], -t*matrix_elem), dim=-1)
    )


def local_estimator(samples, obs, wf_ansatz, batchsum=False):
    """
        Parameters:
        -----------
        ACCEPTS BATCHED SAMPLES

        samples: binary array 
            A batch of occupation number states, e.g. [[1,0,0,1]].
        obs: Function representing the action of the Hamiltonian (or any other observable `obs`
            that is to be measured in VMC) onto the basis state `sample`. 
            It should return an array of matrix elements and an array of basis states which 
            are connected to the basis state `sample` via the action of the Hamiltonian or observable. 
        wf_ansatz: Wavefunction ansatz object which is defined as providing 
            the methods 
                wf_ansatz.psi_amplitude_I(sample_I)
                wf_ansatz.psi_amplitude(sample)
            where `sample_I` is a batch of bitcoded integers and `sample` is a batch 
            of occupation number states. 
        batchsum: boolean
            Whether to sum the local estimator over batches, which results in a scalar output,
            or not, which results in a batched output. 

        Returns:
        --------
        local_estimator: float 
            Value of the local estimator (*summed over all batches* for batchsum=True), e.g. for 
            the local energy with Hamiltonian operator `H`

            E_loc = \sum_b \sum_i \frac{ < b | H | i > < i | \psi > }{ < b | psi > }

    """    
    samples = torch.as_tensor(samples)
    ns = samples.shape[-1]
    nsamples = samples.shape[0] # first dimension is batch dimension
    num_particles = np.count_nonzero(samples[0,...])
    assert ns == wf_ansatz.D, "wf_ansatz.D=%d, ns=%d"%(wf_ansatz.D, ns)
    assert(np.all( np.count_nonzero(samples, axis=1) == num_particles))

    connecting_states, matrix_elements = obs( samples )

    amps_cs = wf_ansatz.psi_amplitude_I(connecting_states[...])
    amp = wf_ansatz.psi_amplitude_unbatch(samples)

    if batchsum:
        sss = 'bi,bi,b ->'
    else:
        sss = 'bi,bi,b -> b'
    local_estimator = torch.einsum(sss, matrix_elements, amps_cs, 1.0/amp)

    if batchsum:
        local_estimator /= float(nsamples) # average over batch

    return local_estimator