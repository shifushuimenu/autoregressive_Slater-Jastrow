import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from physics import *
from slater_sampler_ordered import *
from utils import *
from bitcoding import *


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

        
def kinetic_term( I, lattice ):
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
    hop_from_to = []
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
            hop_from_to.append((i,j))

    # Set states in `I_prime` which were annihilated by the hopping operator to a valid value (i.e.
    # in the correct particle number sector) so that downstream subroutines don't crash. 
    # This does not introduce errors, since the corresponding matrix elements are zero anyways
    # so that their multiplicative contribution is zero. 
    config = int2bin(I, ns=ns)
    num_particles = np.count_nonzero(config[0])    
    I_prime[I_prime == 0] = 2**num_particles - 1

    return ( hop_from_to, I_prime ,  matrix_elem )
               

def tVmodel_loc(config, psi_func, psi_loc, V=0.0):
    '''
    Local energy of periodic 1D t-V model
    
    Args:
       config (1D array): occupation numbers as bitstring.
       psi_func (func): wave function
       psi_loc (number): projection of wave function onto config <config|psi>
       V (float): nearest-neighbout interaction

    Returns:
       number: local energy <config|H|psi> / <config|psi>
    '''
    config = np.array(config)
    nsites = len(config[-1])
    I = bin2int(config)
    lattice = Lattice1d(ns=nsites)
    hop_from_to, connecting_states_I, kin_matrix_elements = kinetic_term([I], lattice)
    connecting_states = int2bin(connecting_states_I, ns=nsites)

    wl, states = [], []
    # nearest neighbour interactions
    nn_int = np.roll(config,-1) * config
    wl.append(V* (nn_int).sum(axis=-1).item())
    states.append(config)

    for ss, mm in zip(connecting_states[0], kin_matrix_elements[0]):
        wl.append(mm)
        states.append(ss[None,:]) # Note: ansatz.psi requires batch dim

    acc = 0.0
    for wi, config_i, (r,s) in zip(wl, states, hop_from_to):
        # eng_i = wi * self.psi_ratio(r,s)
        eng_i = wi * (psi_func(config_i) / psi_loc) 
        acc += eng_i
    return acc

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

    def prob(self,config):
        '''
        probability of configuration.

        Args:
           config (1darray): the bit string as a configuration 

        Returns:
           number: probability |<config|psi>|^2
        '''
        config = np.array(config)
        return abs(self.ansatz.psi(torch.from_numpy(config)).item())**2 

    def local_measure(self, config):
        '''
        get local quantities energy_loc, grad_loc.

        Args:
           config (1darray): the bit string as a configuration.

        Returns:
           number, list: local energy and local gradient for variables. 
        '''
        config = np.array(config)
        psi_loc = self.ansatz.psi(torch.from_numpy(config))

        with torch.autograd.set_detect_anomaly(True):
            # get gradient {d/dW}_{loc}
            self.ansatz.zero_grad()
            psi_loc.backward()
        grad_loc = [p.grad.data/psi_loc.item() for p in self.ansatz.parameters()]

        # E_{loc}
        eloc = self.energy_loc(config, lambda x: self.ansatz.psi(torch.from_numpy(x)).data, psi_loc.data).item()
        return eloc, grad_loc


class SlaterJastrow(nn.Module):
    '''
    Simple Slater-Jastrow ansatz.
    '''
    def __init__(self, eigfunc, num_particles):
        super(SlaterJastrow, self).__init__()
        self.eigfunc = torch.Tensor(eigfunc).double()
        self.num_particles = num_particles 
        self.D = self.eigfunc.shape[0]
        assert self.num_particles <= self.D
        self.Sdet = SlaterDetSampler_ordered(self.eigfunc, self.num_particles)

    def psi(self, config):
        "first dim of config is batch dimension"
        config = np.array(config)
        assert len(config.shape) == 2 and config.shape[0] == 1
        return self.Sdet.psi_amplitude(config[0]) # remove batch dim 

    def psi_I(self, config_I):
        "input is bitcoded integer"
        config = int2bin(config_I, self.D)
        return self.psi(config)

    def prob(self, config):
        config = np.array(config)
        return abs(self.psi(torch.from_numpy(config)).item())**2 

    def sample(self, initial_config, num_bath, num_sample):
        '''
        obtain a set of samples.

        Args:
            initial_config (1darray): initial configuration.
            num_bath (int): number of updates to thermalize.
            num_sample (int): number of samples.

        Return:
            list: a list of spin configurations.
        '''
        print_step = np.Inf # steps between two print of accept rate, Inf to disable showing this information.

        sample, prob_sample = self.Sdet.sample()
        initial_config = list(sample)

        config = initial_config
        prob = self.prob([config])

        n_accepted = 0
        sample_list = []
        for i in range(num_bath + num_sample):
            # generate new config (via direct sampling from Slater determinant)
            # and calculate probability ratio
            with torch.no_grad():
                sample, prob_sample = self.Sdet.sample()
            config_proposed = list(sample)
            prob_proposed = self.prob([config_proposed])
            print("prob=", prob, "prob_sample=", prob_sample, "prob_proposed=", prob_proposed)

            # accept/reject a move by Metropolis algorithm 
            if np.random.random() < prob_proposed / prob:
                config = config_proposed 
                prob = prob_proposed 
                n_accepted += 1

            # print statistics 
            if i % print_step == print_step - 1:
                print('%-10s Accept rate: %.3f' %
                    (i + 1, n_accepted * 1. / print_step))
                n_accepted = 0

            # add sample 
            if i >= num_bath:
                sample_list.append(config_proposed)
                    
        return sample_list


if __name__ == "__main__":

    torch.set_default_dtype(default_dtype_torch)

    from test_suite import *
    (nsites, U) = prepare_test_system_zeroT(5)
    print(U)
    SJ = SlaterJastrow(U, 3)
    sample_list = SJ.sample(initial_config=[0,1,1,0,1], num_bath=20, num_sample=100)
    print("SJ.sample()", sample_list)
    config = [[0,1,1,0,1]]
    print(SJ.psi(config))
    print(SJ.psi_I([7]))

    for name, p in SJ.named_parameters():
        print("name", name, "-> param", p)


    VMC = VMCKernel(tVmodel_loc, SJ)
    print(VMC.prob(config))
    print(VMC.local_measure(config))

    sample_list = SJ.sample(initial_config=[0,1,1,0,1], num_bath=20, num_sample=100)

    energy, gradient_mean, energy_gradient, energy_precision = vmc_measure(
        VMC.local_measure, sample_list, num_bin=50)