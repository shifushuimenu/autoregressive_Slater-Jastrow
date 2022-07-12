import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from physics import *
from slater_sampler_ordered import *
#from slater_sampler_ordered_memory_layout import SlaterDetSampler_ordered
from utils import *
from one_hot import occ_numbers_collapse
from bitcoding import *

from test_suite import local_OBDM, ratio_Slater

from SlaterJastrow_ansatz import SlaterJastrow_ansatz
from k_copy import sort_onehop_states

from torchviz import make_dot

#from profilehooks import profile
from time import time 


###############################
# Just for testing purposes

class PhysicalSystem(object):
    """holds system parameters such as lattice, interaction strength etc.
       for a hypercubic system"""
    def __init__(self, nx, ny, ns, num_particles, D, Vint):
        self.nx = nx; self.ny = ny
        assert ns == nx*ny
        self.ns = nx*ny
        assert num_particles <= ns
        self.np = num_particles
        if ny == 1: assert D == 1
        assert not nx == 1
        self.Vint = Vint
        if D == 1:
            self.lattice = Lattice1d(ns=self.nx)
        elif D == 2:  
            self.lattice = Lattice_rectangular(self.nx, self.ny)    


    def local_energy_slow(self, config, psi_loc, ansatz):
        """Recalculate |<psi|beta>|^2 for each beta."""
        config = np.array(config).astype(int)
        assert len(config.shape) > 1 and config.shape[0] == 1 # just one sample per batch
        nsites = len(config[0])
        I = bin2int_nobatch(config[0])

        hop_from_to, onehop_states_I, kin_matrix_elements = sort_onehop_states(*kinetic_term2(I, self.lattice))
        onehop_states = int2bin(onehop_states_I, ns=nsites)

        wl, states, from_to = [], [], []

        # diagonal matrix element: nearest neighbour interactions
        Enn_int = 0.0
        config_2D = config[0].reshape((self.nx, self.ny))
        for nd in range(self.lattice.coord // 2):
            Enn_int += self.Vint * ( np.roll(config_2D, shift=-1, axis=nd) * config_2D ).sum()      
        wl.append(Enn_int)
        states.append(config)
        from_to.append((0, 0)) # diagonal matrix element: no hopping => choose r=s=0 by convention

        for ss, mm, rs_pair in zip(onehop_states, kin_matrix_elements, hop_from_to):
            wl.append(mm)
            states.append([ss]) # Note: ansatz.psi requires batch dim
            from_to.append(rs_pair)

        assert len(from_to) == len(states) == len(wl)
        OBDM_loc = local_OBDM(alpha=config[0], sp_states = ansatz.slater_sampler.P_ortho.detach().numpy())
        acc = 0.0
        abspsi = [] # amplitudes of all connecting states 
        for wi, config_i, (r,s) in zip(wl, states, from_to):
            if not (r==0 and s==0):                
                # The repeated density estimation of very similar configurations is the bottleneck. 
                abspsi_conf_i = torch.sqrt(ansatz.prob(config_i)).item() 
                abspsi.append(abspsi_conf_i)
                # IMPROVE: Calculate sign() of ratio of Slater determinant directly from the number of exchanges 
                # that brings one state to the other. (Is this really correct ?)
                ratio = (abspsi_conf_i / abs(psi_loc)) * np.sign(ratio_Slater(OBDM_loc, alpha=config[0], beta=config_i[0], r=r, s=s))
            else:
                ratio = 1.0 # <alpha/psi> / <alpha/psi> = 1

            eng_i = wi * ratio

            # ==============================================
            # assert np.isclose( (psi_func(config_i) / psi_loc), ratio ), "Error: ratio1= %16.8f, ratio2 = %16.8f" % (psi_func(config_i) / psi_loc, ratio)
            # Alternative approach:
            # Recalculate wave function aplitude for each connecting state 
            # without using low-rank update. 
            # eng_i = wi * (psi_func(config_    i) / psi_loc) 
            # ==============================================
            acc += eng_i

        return acc, abspsi


    #@profile
    def local_energy(self, config, psi_loc, ansatz, lowrank_flag=True):
        '''
        Local energy of periodic 1D or 2D t-V model
        
        Args:
        config (1D array): occupation numbers as bitstring.
        psi_func (func): wave function amplitude
        psi_loc (number): projection of wave function onto config <config|psi>
        V (float): nearest-neighbout interaction

        Returns:
        number: local energy <config|H|psi> / <config|psi>
        '''
        config = np.array(config).astype(int)
        assert len(config.shape) > 1 and config.shape[0] == 1 # just one sample per batch

        if lowrank_flag:
            # diagonal matrix element: nearest neighbour interactions
            Enn_int = 0.0
            config_2D = config[0].reshape((self.nx, self.ny))

            for nd in range(self.lattice.coord // 2):
                Enn_int += ( np.roll(config_2D, shift=-1, axis=nd) * config_2D ).sum() 

            I = bin2int_nobatch(config[0])
            E_kin_loc, b_absamp = ansatz.lowrank_kinetic(I_ref=I, psi_loc=psi_loc, lattice=self.lattice)
            # # REMOVE
            # E_tot_slow, abspsi = self.local_energy_slow(config, psi_loc, ansatz)
            # print("=========================")
            # print("E_tot_slow=", E_tot_slow)
            # print("E_tot_fast=", E_kin_loc + self.Vint * Enn_int )
            # print("E_kin_loc=", E_kin_loc)
            # print("=========================")
            # # REMOVE
            return E_kin_loc + self.Vint * Enn_int 
        else:
            E_tot_slow, abspsi = self.local_energy_slow(config, psi_loc, ansatz)
            return E_tot_slow
            # E_tot_slow
            #print("E_tot_lowrank=", E_kin_loc + self.Vint * Enn_int)
            # E_tot_slow, abspsi = self.local_energy_slow(config, psi_loc, ansatz)
            #print("E_tot_slow=", E_tot_slow)
            #print("b_absamp=", b_absamp)
            #print("abspsi=", abspsi)
        
        


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
    I = np.array(I, dtype='object') 
    neigh = lattice.neigh
    ns = lattice.ns
    coord = lattice.coord 
    max_num_connect = ns*(coord//2)
    
    I_prime = np.empty(I.shape + (max_num_connect,), dtype='object')
    matrix_elem = np.empty_like(I_prime)
    hop_from_to = []
    count = -1
    for d in range((coord//2)): ####### Replace this error-prone hack by a sum over hopping-bonds. 
        for i in range(ns):     #######
            M = np.zeros_like(I, dtype='object')
            count += 1 
            j = neigh[i, d]
            M[...] = pow(2, i) + pow(2, j)
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
def kinetic_term2( I, lattice, t_hop=1.0 ):
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
            I_prime: list of ints of length `max_num_connect`
                List of states connected to I by the application of the kinetic operator K_kin.
                `max_num_connect` is the number of distinct hopping terms in the kinetic
                operator. If a given hopping term annihilates state |I>, the "connecting state" is still recorded, 
                however with matrix element zero. This is to ensure that, given a batch of samples, 
                            
            matrix_elem: list of floats             
                <I| K_kin |I_prime> for all possible I_prime given the lattice structure. 

        Example:
        --------
    """
    #I = np.asarray(I, dtype='object')
    assert type(I) == int 
    neigh = lattice.neigh
    ns = lattice.ns
    coord = lattice.coord

    # preallocate
    max_num_connect = ns*(coord//2)  # for cubic lattice   
    I_prime = np.empty((max_num_connect,), dtype='object')
    matrix_elem = np.empty_like(I_prime)
    rs_pos = [] # particle hopping from position r to position s

    count = 0
    for d in range((coord//2)): ####### Replace this error-prone hack by a sum over hopping-bonds. 
        for i in range(ns):     #######
            j = neigh[i, d]
            pow2i = 1 << i; pow2j = 1 << j # 2**i and 2**j
            M = pow2i + pow2j
            K = M & I # bitwise AND
            L = K ^ M # bitwise XOR
            STATE_EXISTS = ((K != 0) & (L != 0) & (L != K))
            if STATE_EXISTS:
                I_prime[count] = I - K + L
                ii = min(i,j)
                jj = max(i,j)
                matrix_elem[count] = -t_hop * fermion_parity2(ns, I, ii, jj)

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



#@profile
def vmc_measure(local_measure, sample_list, log_probs, precond, num_bin=50):
    '''
    get energy, gradient and there product averaged over a batch of samples 

    Args:
        local_measure (func): local measurements function, input configuration, return local energy and local gradient.
        sample_list (list): a list of configurations.
        log_probs (lit): the log probs for the configurations.
        num_bin (int): number of bins in binning statistics.

    Returns:
        tuple: expectation valued of energy, gradient, energy*gradient and error of energy.
    '''
    print("inside VMC measure")
    # measurements
    energy_loc_list, grad_loc_list = [], []
    for i, (config, log_prob) in enumerate(zip(sample_list, log_probs)):
        print("local energy: sample nr=", i)
        # back-propagation is used to get gradients.
        energy_loc, grad_loc = local_measure([config], log_prob) # ansatz.psi requires batch dim)
        energy_loc_list.append(energy_loc)
        grad_loc_list.append(grad_loc)


        precond.accumulate(grad_loc)


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
       phys_system: class containing lattice parameters, interaction strengths etc. 
       energy_loc (func): local energy <x|H|\psi>/<x|\psi>.
       ansatz (Module): torch neural network
    '''
    def __init__(self, energy_loc, ansatz):
        self.ansatz = ansatz
        self.energy_loc = energy_loc

        self.t_psiloc = 0 # total time for calculating psi_loc for gradients and local energy
        self.t_sampling = 0 # total time for generating samples
        self.t_grads = 0 # total time for calculating gradients 
        self.t_locE = 0  # total time for calculating local energy 
        self.t_SR = 0    # total time for stochastic reconfiguration (lazy construction of Fisher information matrix S, apply S^{-1} using conjugate gradient)

        self.tmp_cnt = -1

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
    def local_measure(self, config, log_prob):
        '''
        get local quantities energy_loc, grad_loc.

        Args:
           config (1darray): the bit string as a configuration.
           log_prob : the log prob of the configuration 

        Returns:
           number, list: local energy and local gradient for variables. 
        '''
        self.tmp_cnt += 1
        config = np.array(config)
        assert len(config.shape) == 2 and config.shape[0] == 1 # Convention: batch dimension required, but only one sample per batch allowed
        t0 = time()
        psi_loc = self.ansatz.psi_amplitude(torch.from_numpy(config))
        t1 = time()
        self.t_psiloc += (t1-t0)
        assert(psi_loc.requires_grad)

        # # Co-optimizing the Slater determinant converts a simple computation graph into a mess,
        # # as can be seen using visualization with torchviz.
        # if self.tmp_cnt  == 1:
        #     viz_graph = make_dot(psi_loc)
        #     viz_graph.view()
        #     exit(1)

        # =================================================================================================
        #  Comment on `backward(retain_graph = True)`
        # =================================================================================================
        # For each configuration that is passed through MADE the computational
        # graph is built anew as the computations are performed (dynamic computation graph in pytorch). 
        # Computations involving the Slater determinant, such as the orbital rotation
        # from the Hartree-Fock determinant to an optimized set of orbitals, do not depend on 
        # specific configurations and therefore are only performed when initializing the slater_sampler
        # module. Therefore, these parts of the computation graph are missing after the computation graph 
        # has been free after calling backward() for the first time. This is why `backward(retain_graph=True)`
        # is necessary.
        # =================================================================================================

        with torch.autograd.set_detect_anomaly(True):
            # get gradient {d/dW}_{loc}
            self.ansatz.zero_grad()
            # if self.ansatz.slater_sampler.optimize_orbitals:
            #     retain_graph = False  # `retain_graph = True` causes an enormous slowdown !
            # else:
            #     retain_graph = False 
            psi_loc.backward(retain_graph=False) # `retain_graph = True` appears to be necessary (only for co-optimization of SlaterDet) because saved tensors are accessed after calling backward()
        grad_loc = [p.grad.data/psi_loc.item() for p in self.ansatz.parameters()]


        t2 = time()
        self.t_grads += (t2-t1)
        # E_{loc}
        t3 = time()
        eloc = self.energy_loc(config, psi_loc.data, ansatz=self.ansatz).item()
        t4 = time()
        self.t_locE += (t4-t3)
        return eloc, grad_loc


def _test():
    import doctest 
    doctest.testmod(verbose=True)

if __name__ == "__main__":

    torch.set_default_dtype(default_dtype_torch)
    from test_suite import *

    _test()

    Nparticles = 8
    Nsites = 16
    num_samples = 1000
    (_, eigvecs) = prepare_test_system_zeroT(Nsites=Nsites, potential='none')

    # Aggregation of MADE neural network as Jastrow factor 
    # and Slater determinant sampler. 
    Sdet_sampler = SlaterDetSampler_ordered(Nsites=Nsites, Nparticles=Nparticles, single_particle_eigfunc=eigvecs)
    SJA = SlaterJastrow_ansatz(slater_sampler=Sdet_sampler, num_components=Nparticles, D=Nsites, net_depth=2)

    sample_list = np.zeros((num_samples, Nsites)) # a numpy array
    log_probs = np.zeros((num_samples,))
    for i in range(num_samples):
        sample_unfolded, log_prob_sample = SJA.sample_unfolded()
        sample_list[i] = occ_numbers_collapse(sample_unfolded, Nsites).numpy()
        log_probs[i] = log_prob_sample
        print("amplitude=", SJA.psi_amplitude([sample_list[i]]))
        exit(1)
    print("SJ.sample()", sample_list)

    print(SJA.psi_amplitude([[0,1,0,0,1]]))
    print(SJA.psi_amplitude_I([9]))

    #for name, p in SJA.named_parameters():
    #    print("name", name, "-> param", p)

    phys_system = PhysicalSystem(nx=Nsites, ny=1, ns=Nsites, np=Nparticles, D=1, Vint=5.0)

    VMC = VMCKernel(energy_loc=phys_system.local_energy, ansatz=SJA)
    print(VMC.prob([[0,1,0,0,1]]))
    print("VMC.local_measure")
    e_loc, grad_loc = VMC.local_measure([[0,1,0,0,1]])

    energy, gradient_mean, energy_gradient, energy_precision = vmc_measure(
        VMC.local_measure, sample_list, log_probs, num_bin=10)

