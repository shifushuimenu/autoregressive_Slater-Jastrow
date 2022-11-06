# TODO: standardized interface for PhysicalSystem and Ansatz
import torch
import numpy as np

from physics import Lattice1d, Lattice_rectangular, kinetic_term
from bitcoding import bin2int_nobatch, int2bin

from slater_determinant import local_OBDM, ratio_Slater

from k_copy import sort_onehop_states

# from torchviz import make_dot

#from profilehooks import profile
from time import time 

class PhysicalSystem(object):
    """
    Class holds system parameters such as lattice, interaction strength etc.
    for a hypercubic system
    and provides method for calculating local energy.
    """
    def __init__(self, nx, ny, ns, num_particles, dim, Vint):
        self.nx = nx; self.ny = ny
        assert ns == nx*ny
        self.ns = nx*ny
        assert num_particles <= ns
        self.np = num_particles
        if ny == 1: assert D == 1
        assert not nx == 1
        self.Vint = Vint
        if dim == 1:
            self.lattice = Lattice1d(ns=self.nx)
        elif dim == 2:  
            self.lattice = Lattice_rectangular(self.nx, self.ny)    

    #@profile
    def local_energy(self, config, psi_loc, ansatz, lowrank_flag=True):
        """
        Local energy of periodic 1D or 2D t-V model
        
        Args:
            config (1D array): occupation numbers as bitstring.
            psi_loc (number): projection of wave function onto config <config|psi>
            ansatz: (object): wavefunction ansatz providing function lowrank_kinetic() 
            V (float): nearest-neighbout interaction

        Returns:
            number: local energy <config|H|psi> / <config|psi>
        """
        config = np.array(config).astype(int)
        assert len(config.shape) > 1 and config.shape[0] == 1 # just one sample per batch

        if lowrank_flag:
            # diagonal matrix element: nearest neighbour interaction
            config_2D = config[0].reshape((self.nx, self.ny))
            Enn_int = 0.0
            for nd in range(self.lattice.coord // 2):
                Enn_int += ( np.roll(config_2D, shift=-1, axis=nd) * config_2D ).sum() 

            I = bin2int_nobatch(config[0])
            E_kin_loc, b_absamp = ansatz.lowrank_kinetic(I_ref=I, psi_loc=psi_loc, lattice=self.lattice)
            return E_kin_loc + self.Vint * Enn_int 
        else:
            E_tot_slow, abspsi = self._local_energy_slow(config, psi_loc, ansatz)
            return E_tot_slow        


    def _local_energy_slow(self, config, psi_loc, ansatz):
        """
        Recalculate |<psi|beta>|^2 for each beta
        """
        config = np.array(config).astype(int)
        assert len(config.shape) > 1 and config.shape[0] == 1 # just one sample per batch
        nsites = len(config[0])
        I = bin2int_nobatch(config[0])

        hop_from_to, onehop_states_I, kin_matrix_elements = sort_onehop_states(*kinetic_term(I, self.lattice))
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
            states.append([ss]) # ansatz.psi requires batch dim
            from_to.append(rs_pair)

        assert len(from_to) == len(states) == len(wl)
        OBDM_loc = local_OBDM(alpha=config[0], sp_states = ansatz.slater_sampler.P_ortho.detach().numpy())
        acc = 0.0
        abspsi = [] # amplitudes of all connecting states 
        for wi, config_i, (r,s) in zip(wl, states, from_to):
            if not (r==0 and s==0):                
                # The repeated density estimation of very similar configurations is the bottleneck. 
                abspsi_conf_i = np.sqrt(ansatz.prob(config_i).item()) 
                abspsi.append(abspsi_conf_i)
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

        
class VMCKernel(object):
    """
    variational Monte Carlo kernel.

    Args:
       phys_system: class containing lattice parameters, interaction strengths etc. 
       energy_loc (func): local energy <x|H|\psi>/<x|\psi>.
       ansatz (Module): torch neural network
    """
    def __init__(self, energy_loc, ansatz):
        self.ansatz = ansatz
        self.energy_loc = energy_loc # function(config, psi_loc.data, ansatz)

        self.t_psiloc = 0 # total time for calculating psi_loc for gradients and local energy
        self.t_sampling = 0 # total time for generating samples
        self.t_grads = 0 # total time for calculating gradients 
        self.t_locE = 0  # total time for calculating local energy 
        self.t_SR = 0    # total time for stochastic reconfiguration (lazy construction of Fisher information matrix S, apply S^{-1} using conjugate gradient)

        self.tmp_cnt = -1

    #@profile
    def prob(self,config):
        """
        probability of configuration.

        Args:
           config (1darray): the bit string as a configuration 

        Returns:
           number: probability |<config|psi>|^2
        """
        config = np.asarray(config)
        return self.ansatz.prob(config)

    #@profile
    def local_measure(self, config, log_prob):
        """
        get local energy and gradients 

        Args:
           config (1darray): the bit string as a configuration.
           log_prob : the log prob of the configuration 

        Returns:
           number, list: local energy and local gradient for variables (as numpy values)
        """
        self.tmp_cnt += 1
        config = np.asarray(config)
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
        # has been freed after calling backward() for the first time. This is why `backward(retain_graph=True)`
        # is necessary unless the orbital rotation is repeated after each backpropagation. We do the latter 
        # so that ew can use `backward(retain_graph=False)`, which is faster. 
        # =================================================================================================

        with torch.autograd.set_detect_anomaly(True):
            # get gradient {d/dW}_{loc}
            self.ansatz.zero_grad()
            psi_loc.backward(retain_graph=False)
        grad_loc = [p.grad.data/psi_loc.item() for p in self.ansatz.parameters()]

        t2 = time()
        self.t_grads += (t2-t1)
        t3 = time()
        with torch.no_grad():
            eloc = self.energy_loc(config, psi_loc.data, ansatz=self.ansatz).item()
        t4 = time()
        self.t_locE += (t4-t3)

        return eloc, grad_loc


def _test():
    import doctest 
    doctest.testmod(verbose=True)

if __name__ == "__main__":
    _test()
