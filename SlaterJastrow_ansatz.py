"""
   Variational ansatz for a correlated fermionic wavefunction: 
   Single Slater determinant and Jastrow factor realized by an 
   autoregressive neural network.
"""
# TODO: add examples 

import numpy as np 
import torch 
from utils import default_dtype_torch, TEST_FAST_SAMPLING
from torch.distributions.one_hot_categorical import OneHotCategorical

from selfMADE import selfMADE

import os

if TEST_FAST_SAMPLING:
    # CAREFUL: this crashes in long runs because of zero probabilities
    from slater_sampler_ordered_memory_layout import SlaterDetSampler_ordered
else:
    from slater_sampler_ordered import SlaterDetSampler_ordered

from one_hot import occ_numbers_unfold, occ_numbers_collapse
from bitcoding import int2bin, bin2pos
        
# imports needed in lowrank_kinetic()
from k_copy import sort_onehop_states 
from physics import kinetic_term
from slater_determinant import ratio_Slater, local_OBDM

from time import time 
#from profilehooks import profile 

__all__ = ['SlaterJastrow_ansatz']


def _log_cutoff(x):
    """
    Replace -inf by a very small, but finite value.
    """
    return np.where(x > 0, np.log(x), -1000)

def _cond_prob2log_prob(xs_hat, xs_unfolded, xs_pos, num_components, D):
    """
    First dimension is batch dimension. 

    Pick conditional probabilities at actually sampled positions so as to 
    get the probability of the microconfiguration.

    Inputs are conditional probabilities.
    """
    assert xs_hat.shape == xs_unfolded.shape 
    assert len(xs_hat.shape) == 2
    num_states = xs_hat.shape[0]
    mm = xs_unfolded[:,:] * xs_hat[:,:]
    # CAREFUL: this may be wrong if a probability is accidentally zero !
    # introduce a boolean array which indicates the valid support
    # and use log-probabilities throughout.  
    supp = np.empty((num_states, num_components, D), dtype=bool)
    supp[...] = False 
    for l in range(num_states):
        for k in range(num_components):
            xmin = 0 if k==0 else xs_pos[l, k-1] + 1
            xmax = D - num_components + k + 1
            supp[l, k, xmin:xmax] = True
    supp = supp.reshape(-1, num_components*D)
    assert mm.shape == supp.shape
    # CAREFUL
    log_probs = _log_cutoff(np.where(mm > 0, mm, 1.0)).sum(axis=-1)

    return log_probs 


class SlaterJastrow_ansatz(selfMADE):
    """
        Variational wavefunction ansatz for a fermionic many-particle system in the 
        canonical ensemble.

        This is realized by aggregation of MADE neural network, representing the Jastrow factor, and 
        a Slater determinant sampler 'SlaterDetSampler', which is provided by the calling 
        routine, into a fermionic many-body wavefunction. 

        The parameters for 'SlaterDetSampler' and MADE need to be chosen consistently,
        see example below. 

        Parameters:
        -----------
            D: int 
                Dimension of the single-particle Hilbert space. For fermions on a lattice,
                this is typically just the number of lattice sites, When studying Hamiltonians 
                appearing in quantum chemistry, `D` would denote the number of (spin-) orbitals.
            num_components: int
                The number of particles. The fermionic exclusion principle requires `num_components` < `D`. 
            net_depth: int >= 2
                The number of layers of MADE neural network.
                Even if the specified `net_depth` is < 2, there 
                will be at least two autoregressive layers in the neural network.

        Example: 
        --------
        >>> N_particles = 3; N_sites = 5;
        >>> from HF import prepare_test_system_zeroT
        >>> (N_sites, eigvecs) = prepare_test_system_zeroT(Nsites=N_sites);
        >>> SdetSampler = SlaterDetSampler(eigvecs, N_particles);
        >>> SJA = SlaterJastrow_ansatz(SdetSampler, D=N_sites, num_components=N_particles, net_depth=3);
    """
    def __init__(self, slater_sampler, **kwargs):
        self.D = kwargs['D']
        self.num_components = kwargs['num_components']
        self.net_depth = kwargs['net_depth']
        self.deactivate_Jastrow = kwargs.get('deactivate_Jastrow', False)
        print("isinstance(Sdet_Sampler, SlaterDetSampler_ordered)=", isinstance(slater_sampler, SlaterDetSampler_ordered))
        assert isinstance(slater_sampler, SlaterDetSampler_ordered) or slater_sampler is None 
        self.input_orbitals = (slater_sampler is not None) 

        if slater_sampler is not None:
           assert kwargs['num_components'] == slater_sampler.N
           assert kwargs['D'] == slater_sampler.D
        else:         
           slater_sampler = SlaterDetSampler_ordered(Nsites=self.D, Nparticles=self.num_components,
                   single_particle_eigfunc=None, naive_update=True)
        
        slater_sampler.reset_sampler() # make sure sampling can start from the first component
        super(SlaterJastrow_ansatz, self).__init__(**kwargs)

        self.slater_sampler = slater_sampler

        self.t_logprob_B = 0
        self.t_logprob_F = 0
       

    def sample(self):
        sample_unfolded, log_prob_sample = self.sample_unfolded()
        return occ_numbers_collapse(sample_unfolded, self.D)


    def sample_unfolded(self, seed=None):
        """
            Sample particle positions componentwise including, for each component,
            :the probability coming from componentwise sampling of the Slater determinant. 

            The internal state of the Slater determinant sampler is reset so that 
            sampling can start from the first component.

            Parameter
            ---------
                seed: int (optional)
                    Seed the random number generator with a fixed seed (so as to aid 
                    reproducibility during debugging)

            Returns
            -------
                x_out 
                log_prob_sample
        """
        if seed:
            torch.manual_seed(seed)
        with torch.no_grad():
            prob_sample = 1.0 # probability of the generated sample
            log_prob_sample = 1.0
            self.slater_sampler.reset_sampler(rebuild_comp_graph=False) # we do not need the computation graph since we will not backpropagate during sampling
            x_out = occ_numbers_unfold(sample=torch.zeros(self.D).unsqueeze(dim=0), unfold_size=self.num_components,
                duplicate_entries=False)   
            pos_one_hot = torch.zeros(self.D).unsqueeze(dim=0)

            ##fh = open("timing.dat", "a")
            ##t_SD = 0
            ##t_net = 0

            # The particle sampled first should be on the leftmost position. 
            # Pauli_blocker is part of Softmax layer. 
            # x_hat_bias is not affected by the Pauli blocker in the Softmax layer 
            # and is overwritten in selfMADE.__init__().            
            for i in range(0, self.num_components):

                ##t1 = time()
                x_hat = self.forward(x_out)
                ##t2 = time()
                ##t_net += (t2-t1)

                #t1 = time()
                cond_prob_fermi = self.slater_sampler.get_cond_prob(k=i)
                #t2 = time()
                #t_SD += (t2-t1)

                # Make sure that an ordering of the particles is enforced.
                # The position of particle i, k_i, is always to the left of k_{i+1},
                # i.e. k_i
                #  < k_{i+1}.
                probs = x_hat[:,i*self.D:(i+1)*self.D] * cond_prob_fermi
                # With the factor coming from Slater determinant the probabilities are not normalized.                    
                # OneHotCategorical() accepts unnormalized probabilities, still it is better to normalize.
                norm = torch.sum(probs, dim=-1)
                probs = probs / norm

                # Checks 
                assert np.isclose(torch.sum(probs, dim=-1).item(), 1.0)
                ## clamp negative values which are in absolute magnitude below machine precision
                probs = torch.where(abs(probs) > 1e-8, probs, torch.tensor(0.0))
                if i==0:
                    if self.input_orbitals:
                       assert(np.all(np.isclose(probs.numpy(), cond_prob_fermi)))

                pos_one_hot = OneHotCategorical(probs).sample() 
                k_i = torch.nonzero(pos_one_hot[0])[0][0]

                prob_sample *= probs[0, k_i].item()  # index 0: just one sample per batch
                log_prob_sample += np.log(probs[0, k_i].item())

                x_out[:,i*self.D:(i+1)*self.D] = pos_one_hot

                self.slater_sampler.update_state(pos_i = k_i.item())

                # init Pauli blocker/orderer for next pass 
                if i < self.num_components-1:
                    self.net[-1].Pauli_blocker[i+1,:] = 0.0
                    self.net[-1].Pauli_blocker[i+1,:k_i+1] = torch.tensor([float('-inf')]) 
                    self.net[-1].Pauli_blocker[i+1,self.D-self.num_components+(i+1)+1:] = torch.tensor([float('-inf')])

        # reset (to be sure)
        # Don't reset the zero-th component, it has been set in selfMADE.__init__().
        self.net[-1].Pauli_blocker[1:,:] = 0.0

        # check the probability of the generated basis state 
        # log_prob = self.log_prob(occ_numbers_collapse(x_out, self.D))
        # print("log_prob.requires_grad=", log_prob.requires_grad)
        # assert(log_prob.requires_grad)

        ##print("## t_SD=%10.6f, t_net=%10.6f" %(t_SD, t_net), file=fh)
        ##fh.close()

        return x_out, log_prob_sample

    #@profile 
    def log_prob(self, samples):
        """
            Logarithm of the amplitude squared of the wave function on an 
            occupation number state. 

            Input:
            ------
                samples: binary array, i.e. 
                    torch.Tensor([[1,0,1,0], [0,0,1,1], [1,1,0,0]])
                    First dimension is batch dimension. 
        """
        samples = torch.tensor(np.asarray(samples))
        assert samples.shape[0] == 1 # just one sample per batch

        # The conditional fermionic probabilities depend on the particle positions
        # and need to be recalculated iteratively.
        # Therefore batch processing is not possible. 
        pos = bin2pos(samples)[0]

        # During density estimation the Pauli blocker layer is not needed. 
        with torch.no_grad():
           self.net[-1].Pauli_blocker[:,:] = 0.0

        samples_unfold = occ_numbers_unfold(samples, duplicate_entries=False)
        # Flatten leading dimensions (This is necessary since there may be a batch of samples 
        # for several "connecting states", but forward() accepts only one batch dimension.)
        samples_unfold_flat = samples_unfold.view(-1, samples_unfold.shape[-1])

        if self.deactivate_Jastrow:
            # for debugging 
            x_hat_B = torch.ones_like(samples_unfold_flat, requires_grad=False)
        else:
            # for MADE a single forward pass gives all probabilities 
            t1 = time()
            x_hat_B = self.forward(samples_unfold_flat)
            t2 = time()
            self.t_logprob_B += (t2-t1)

        x_hat_F = torch.zeros_like(x_hat_B, requires_grad=False)
        self.slater_sampler.reset_sampler()
        # for Slater sampler we need to go through all sampling steps because it has an internal state,
        # which needs to be updated 
        t1 = time()
        for k in range(0, self.num_components): 
            #t11 = time()
            x_hat_F[..., k*self.D:(k+1)*self.D] = self.slater_sampler.get_cond_prob(k=k)
            #t22 = time()
            #print("get cond prob k-th", k, "elapsed=", t22-t11)
            self.slater_sampler.update_state(pos[k].item())
        x_hat = x_hat_B * x_hat_F
        x_hat[..., 0:self.D] = x_hat_F[..., 0:self.D]  # The Jastrow factor does not affect the first component, which is unconditional. 
        t2 = time()
        self.t_logprob_F += (t2-t1)
        for k in range(self.num_components):
            norm = torch.sum(x_hat[..., k*self.D:(k+1)*self.D])
            x_hat[..., k*self.D:(k+1)*self.D] /= norm

        if not self.deactivate_Jastrow:
            assert(x_hat_B.requires_grad)
        if not self.input_orbitals:
            assert(x_hat_F.requires_grad)

        mm = x_hat * samples_unfold_flat # Pick only the probabilities at actually sampled positions !
        ones = torch.ones(*mm.shape)
        log_prob = torch.log(torch.where(mm > 0, mm, ones)).sum(dim=-1)
        # reshape leading dimensions back to original shape (last dim is missing now)
        return log_prob.view(*samples.shape[:-1])            

    #@profile
    def psi_amplitude(self, samples):
        """
            Wavefunction amplitude on an occupation number state. 
            The amplitude can be negative due to a negative contribution
            from the Slater determinant. 

            This quantity is used in the local estimator of observables which are off-diagonal
            in the occupation number basis.

            Input:
            ------
                samples: binary array, i.e. 
                    torch.Tensor([[1,0,1,0], [0,0,1,1], [1,1,0,0]])
                    First dimension is batch dimension. 
        """
        samples = torch.tensor(np.asarray(samples))
        assert len(samples.shape) == 2 and samples.shape[0] == 1 # Convention: just one sample per batch allowed
        amp_abs = torch.sqrt(torch.exp(self.log_prob(samples)))
        amp_sign = torch.sign(self.slater_sampler.psi_amplitude(samples))
        #assert amp_sign.requires_grad
        return amp_abs * amp_sign


    def prob(self, samples):
        samples = torch.tensor(np.asarray(samples))
        assert samples.shape[0] == 1 # only one sample per batch
        return torch.exp(self.log_prob(samples))


    def psi_amplitude_I(self, samples_I):
        """
            Wavefunction amplitude of an occupation number state. 
            This is a wrapper function around `self.psi_amplitude(samples)`.

            Input:
            ------
                samples_I: integer representation I of a binary array.
                    First dimension is batch dimension. 
        """
        # IMPROVE: check for correct particle number 
        samples_I = torch.as_tensor(samples_I)
        assert len(samples_I.shape) >= 1, "Input should be bitcoded integer (with at least one batch dim.)."
        samples = int2bin(samples_I, self.D)
        return self.psi_amplitude_unbatch(samples)


    def lowrank_kinetic(self, I_ref, psi_loc, lattice):        
        """Compute local kinetic energy for a basis state. 

        Input:
        ------
            I_ref : int (no batch dimension)       


        Except in the input to net.forward(), where a torch tensor is used, numpy arrays 
        are used throughout since it is not necessary to backpropagate the kinetic energy.  

                          \sum_b < a | H_kin | b > sqrt(| < b | \psi > |^2) * sign(< b | psi > / < a | psi >)
        E_kin_loc(a)  =  -------------------------------------------------------------------------------------
                                                sqrt(| < a | \psi > |^2)

        """
        rs_pos, xs_I, matrix_elem = sort_onehop_states(*kinetic_term(I_ref, lattice))

        # conditional probabilities of all onehop states
        xs_hat_F, cond_prob_ref_F = self.slater_sampler.lowrank_kinetic(I_ref, xs_I, rs_pos, print_stats=True)        
        ## IMPROVE: Check that the probability amplitude (!) calculated from the conditional 
        ## probabilities `cond_prob_ref` conincides with `psi_loc`. -> assert 
        #pos_ref = bin2pos([config_ref])
        #config_ref_unfolded = occ_numbers_unfold([config_ref])
        #log_prob_ref = _cond_prob2log_prob(xs_hat=cond_prob_ref[None,:], xs_unfolded=config_ref_unfolded, xs_pos=pos_ref) # requires batch dimension 
        #print("log_prob_ref=", log_prob_ref, "psi_loc=", psi_loc, "np.exp(0.5*log_prob_ref)=", np.exp(0.5*log_prob_ref))
        #assert np.exp(0.5*log_prob_ref) == abs(psi_loc)
        
        xs = int2bin(xs_I, ns=self.D)
        xs_unfolded = occ_numbers_unfold(xs, duplicate_entries=False) # output of occ_numbers_unfold() is a torch tensor 
        xs_hat_B = np.empty_like(xs_hat_F)

#        for i, x_unfolded in enumerate(xs_unfolded):
#            xs_hat_B[i, :] = self.forward(x_unfolded.unsqueeze(dim=0)).squeeze(dim=0).numpy() # net.forward() requires batch dimension 
        xs_hat_B[:,:] = self.forward(xs_unfolded).detach().numpy() # process a batch of samples at once

        xs_hat = xs_hat_F[:, :] * xs_hat_B[:, :]
        # normalize: The probabilities x_hat_F and x_hat_B individually are normalized,
        # but their product is not. 
        assert len(xs_hat.shape) == 2 
        for k in range(self.num_components):
            norm = np.sum(xs_hat[:, k*self.D:(k+1)*self.D], axis=1)
            xs_hat[:, k*self.D:(k+1)*self.D] /= norm[:, None] # enable broadcast 

        # Pick only the probabilities at actually sampled positions
        # so as to compute the probability of a sample. 
        xs_unfolded = xs_unfolded.numpy()
        xs_pos = bin2pos(xs)
        log_probs = _cond_prob2log_prob(xs_hat, xs_unfolded, xs_pos, self.num_components, self.D)
        b_absamp = np.exp(0.5*log_probs)

        # sign( <\beta|\psi> / <\alpha|\psi> )
        b_relsign = np.empty_like(b_absamp)
        # local OBDM is needed for computing ratio of two Slater determinants
        config_ref = int2bin(I_ref, ns=self.D)
        OBDM_loc = local_OBDM(alpha=config_ref, sp_states=self.slater_sampler.P_ortho.detach().numpy())
        for i, x in enumerate(xs):            
            (r,s) = rs_pos[i]
            b_relsign[i] = np.sign(ratio_Slater(OBDM_loc, alpha=config_ref, beta=x, r=r, s=s))

        E_kin_loc = np.dot(matrix_elem[:], b_absamp[:] * b_relsign[:]) / abs(psi_loc)

        return E_kin_loc, b_absamp
