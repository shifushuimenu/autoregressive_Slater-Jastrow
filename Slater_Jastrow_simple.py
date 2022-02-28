import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from physics import *
from slater_sampler_ordered import *
from utils import *
from one_hot import occ_numbers_unfold, occ_numbers_collapse
from bitcoding import *

from test_suite import local_OBDM, ratio_Slater

from SlaterJastrow_ansatz import SlaterJastrow_ansatz

from profilehooks import profile

#@profile
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
    """1d lattice with pbc"""
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

        self.neigh = np.vstack((up, right, down, left)).transpose() # idxs: sitenr, direction (up=0, right=1, down=2, left=3)
        
###############################

#@profile        
def kinetic_term( I, lattice, t_hop=1.0 ):
    """
        Parameters:
        -----------
            I: Bitcoded integer representing occupation numbers 
               of spinless fermions. First dimension is batch dimension.
            lattice: Lattice object 
                Provides nearest neighbour matrix which defines the possible 
                hopping terms. 
            t_hop: hopping parameter in 
                 -t_hop \sum_{(i,j) \in bonds} (c_i^{\dagger} c_j + h.c.)
            
        Returns:
        --------
            hop_from_to: list of pairs [(i1_initial, i1_final), (i2_initial, i2_final), ...]
                where state I1_prime is obtained from I[0] by a particle hopping from i1_initial to i1_final, 
                state I2_prime is obtained from I[0] by hopping from i2_initial to i2_final etc. 
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
    hop_from_to = []
    count = -1
    for d in range((coord//2)): ####### Replace this error-prone hack by a sum over hopping-bonds. 
        for i in range(ns):     #######
            M = np.zeros_like(I, dtype=np.int64)
            count += 1 
            j = neigh[i,d]
            M[...] = 2**i + 2**j
            K = np.bitwise_and(M, I)
            L = np.bitwise_xor(K, M)
            STATE_EXISTS = ((K != 0) & (L != 0) & (L != K))
            I_prime[..., count] = np.where(STATE_EXISTS, I - K + L, False)
            ii = min(i,j)
            jj = max(i,j)
            matrix_elem[..., count] = -t_hop * np.where(STATE_EXISTS, fermion_parity(ns, I, ii, jj), 0)
            # hop_from_to.append((i,j))

            # CAREFUL: hop_from_to is only correct if the input is not batched.
            if np.bitwise_and(I, 2**i) == 2**i and np.bitwise_xor(I, I+2**j) == 2**j:
                r = i; s = j
            elif np.bitwise_and(I, 2**j) == 2**j and np.bitwise_xor(I, I+2**i) == 2**i:
                r = j; s = i 
            else:
                r = -1; s = -1
            hop_from_to.append((r,s))

    # Set states in `I_prime` which were annihilated by the hopping operator to a valid value (i.e.
    # in the correct particle number sector) so that downstream subroutines don't crash. 
    # This does not introduce errors, since the corresponding matrix elements are zero anyways
    # so that their multiplicative contribution is zero. 
    config = int2bin(I, ns=ns)
    num_particles = np.count_nonzero(config[0])    
    I_prime[I_prime == 0] = 2**num_particles - 1

    return ( hop_from_to, I_prime, matrix_elem )



#@profile
def tVmodel_loc(config, psi_func, psi_loc, ansatz, V=5.0):
    '''
    Local energy of periodic 1D t-V model
    
    Args:
       config (1D array): occupation numbers as bitstring.
       psi_func (func): wave function amplitude
       psi_loc (number): projection of wave function onto config <config|psi>
       V (float): nearest-neighbout interaction

    Returns:
       number: local energy <config|H|psi> / <config|psi>
    '''
    config = np.array(config)
    assert len(config.shape) > 1 and config.shape[0] == 1 # just one sample per batch
    nsites = len(config[-1])
    I = bin2int(config)
    lattice = Lattice1d(ns=nsites)
    hop_from_to, connecting_states_I, kin_matrix_elements = kinetic_term([I], lattice)
    connecting_states = int2bin(connecting_states_I, ns=nsites)
 
    wl, states, from_to = [], [], []

    # diagonal matrix element: nearest neighbour interactions
    nn_int = V * (np.roll(config, shift=-1) * config).sum(axis=-1).item()
    wl.append(nn_int)
    states.append(config)
    from_to.append((0, 0)) # diagomal matrix element: no hopping => choose r=s=0 by convention

    for ss, mm, rs_pair in zip(connecting_states[0], kin_matrix_elements[0], hop_from_to):
        wl.append(mm)
        states.append(ss[None,:]) # Note: ansatz.psi requires batch dim
        from_to.append(rs_pair)

    acc = 0.0

    assert len(from_to) == len(states) == len(wl)
    print("before local OBDM")
    OBDM_loc = local_OBDM(alpha=config[0], sp_states = ansatz.slater_sampler.P_ortho.detach().numpy())
    print("before density estimation of connecting states")
    for wi, config_i, (r,s) in zip(wl, states, from_to):
        if wi != 0: # there are unphysical connecting states whose matrix elements are set to zero
            if not (r==0 and s==0):
                abspsi_conf_i = torch.sqrt(ansatz.prob(config_i)).item() # The repeated density estimation of very similar configurations is the bottleneck. 
                ratio = (abspsi_conf_i / abs(psi_loc)) * np.sign(ratio_Slater(OBDM_loc, alpha=config[0], beta=config_i[0], r=r, s=s))
            else:
                ratio = 1.0 # <alpha/psi> / <alpha/psi> = 1

            eng_i = wi * ratio

            # ==============================================
            # assert np.isclose( (psi_func(config_i) / psi_loc), ratio ), "Error: ratio1= %16.8f, ratio2 = %16.8f" % (psi_func(config_i) / psi_loc, ratio)
            # Alternative approach:
            # Recalculate wave function aplitude for each connecting state 
            # without using low-rank update. 
            # eng_i = wi * (psi_func(config_i) / psi_loc) 
            # ==============================================
            acc += eng_i
    return acc

#@profile
def vmc_measure(local_measure, sample_list, num_bin=50):
    '''
    perform measurements on samples

    Args:
        local_measure (func): local measurements function, input configuration, return local energy and local gradient.
        sample_list (list): a list of spin configurations.
        num_bin (int): number of bins in binning statistics.

    Returns:
        tuple: expectation valued of energy, gradient, energy*gradient and error of energy.
    '''
    print("inside VMC measure")
    # measurements
    energy_loc_list, grad_loc_list = [], []
    for i, config in enumerate(sample_list):
        # back-propagation is used to get gradients.
        energy_loc, grad_loc = local_measure([config]) # ansatz.psi requires batch dim
        energy_loc_list.append(energy_loc)
        grad_loc_list.append(grad_loc)

    # binning statistics for energy
    energy_loc_list = np.array(energy_loc_list)
    energy, energy_precision = binning_statistics(energy_loc_list, num_bin=num_bin)

    # get expectation values
    energy_loc_list = torch.from_numpy(energy_loc_list)
    if grad_loc_list[0][0].is_cuda: energy_loc_list = energy_loc_list.cuda()
    grad_mean = []
    energy_grad = []
    for grad_loc in zip(*grad_loc_list):
        grad_loc = torch.stack(grad_loc, 0)
        grad_mean.append(grad_loc.mean(0))
        energy_grad.append(
            (energy_loc_list[(slice(None),) + (None,) * (grad_loc.dim() - 1)] * grad_loc).mean(0))
    return energy.item(), grad_mean, energy_grad, energy_precision


def binning_statistics(var_list, num_bin):
    '''
    binning statistics for variable list.
    '''
    num_sample = len(var_list)
    if num_sample % num_bin != 0:
        raise
    size_bin = num_sample // num_bin

    # mean, variance
    mean = np.mean(var_list, axis=0)
    variance = np.var(var_list, axis=0)

    # binned variance and autocorrelation time.
    variance_binned = np.var(
        [np.mean(var_list[size_bin * i:size_bin * (i + 1)]) for i in range(num_bin)])
    t_auto = 0.5 * size_bin * \
        np.abs(np.mean(variance_binned) / np.mean(variance))
    stderr = np.sqrt(variance_binned / num_bin)
    print('Binning Statistics: Energy = %.4f +- %.4f, Auto correlation Time = %.4f' %
          (mean, stderr, t_auto))
    return mean, stderr


class VMCKernel(object):
    '''
    variational Monte Carlo kernel.

    Args:
       energy_loc (func): local energy <x|H|\psi>/<x|\psi>.
       ansatz (Module): torch neural network
    '''
    def __init__(self, energy_loc, ansatz):
        self.ansatz = ansatz
        self.energy_loc = energy_loc

    #@profile
    def prob(self,config):
        '''
        probability of configuration.

        Args:
           config (1darray): the bit string as a configuration 

        Returns:
           number: probability |<config|psi>|^2
        '''
        config = np.array(config)
        return self.ansatz.prob(config)

    #@profile
    def local_measure(self, config):
        '''
        get local quantities energy_loc, grad_loc.

        Args:
           config (1darray): the bit string as a configuration.

        Returns:
           number, list: local energy and local gradient for variables. 
        '''
        config = np.array(config)
        assert len(config.shape) == 2 and config.shape[0] == 1 # Convention: batch dimension required, but only one sample per batch allowed
        psi_loc = self.ansatz.psi_amplitude(torch.from_numpy(config))
        assert(psi_loc.requires_grad)
        with torch.autograd.set_detect_anomaly(True):
            # get gradient {d/dW}_{loc}
            self.ansatz.zero_grad()
            psi_loc.backward(retain_graph=True) # `retain_graph=True` appears to be necessary (?) because saved tensors are accessed after calling backward()
        grad_loc = [p.grad.data/psi_loc.item() for p in self.ansatz.parameters()]

        # E_{loc}
        eloc = self.energy_loc(config, lambda x: self.ansatz.psi_amplitude(torch.from_numpy(x)).data, psi_loc.data, ansatz=self.ansatz).item()
        return eloc, grad_loc

def _test():
    import doctest 
    doctest.testmod(verbose=True)

if __name__ == "__main__":

    torch.set_default_dtype(default_dtype_torch)
    from test_suite import *

    _test()

    Nparticles = 2
    Nsites = 5
    num_samples = 1000
    (_, eigvecs) = prepare_test_system_zeroT(Nsites=Nsites, potential='none')

    # Aggregation of MADE neural network as Jastrow factor 
    # and Slater determinant sampler. 
    Sdet_sampler = SlaterDetSampler_ordered(eigvecs, Nparticles=Nparticles)
    SJA = SlaterJastrow_ansatz(slater_sampler=Sdet_sampler, num_components=Nparticles, D=Nsites, net_depth=2)

    sample_list = np.zeros((num_samples, Nsites)) # a numpy array
    for i in range(num_samples):
        sample_unfolded, sample_prob = SJA.sample_unfolded()
        sample_list[i] = occ_numbers_collapse(sample_unfolded, Nsites).numpy()
        print("amplitude=", SJA.psi_amplitude([sample_list[i]]))
        exit(1)
    print("SJ.sample()", sample_list)

    print(SJA.psi_amplitude([[0,1,0,0,1]]))
    print(SJA.psi_amplitude_I([9]))

    #for name, p in SJA.named_parameters():
    #    print("name", name, "-> param", p)

    VMC = VMCKernel(energy_loc=tVmodel_loc, ansatz=SJA)
    print(VMC.prob([[0,1,0,0,1]]))
    print("VMC.local_measure")
    e_loc, grad_loc = VMC.local_measure([[0,1,0,0,1]])

    energy, gradient_mean, energy_gradient, energy_precision = vmc_measure(
        VMC.local_measure, sample_list, num_bin=10)

