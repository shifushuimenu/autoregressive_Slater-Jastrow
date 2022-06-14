import torch
from torch import nn
import numpy as np
import math

from torch.distributions.categorical import Categorical

from utils import default_dtype_torch

from bitcoding import bin2pos, int2bin

__all__ = ['SlaterDetSampler']


class SlaterDetSampler(torch.nn.Module):
    """
        Sample a set of particle positions from a Slater determinant of 
        orthogonal orbitals via direct componentwise sampling. 
        
        The Slater determinant is constructed from the first `N_particles`
        number of single-particle eigenstates. 
        
        Parameters:
        -----------
        single_particle_eigfunc: 2D arraylike
            Matrix of dimension D x D containing all the single-particle
            eigenfunctions as columns. Note that D is the dimension
            of the single-particle Hilbert space.
        N_particles: int
            Number of particles where N_particles <= D.

        From the matrix containing the single-particle eigenfunctions 
        as columns, the first 'N_particles" columns are chosen to form the 
        Slater determinant.
    """
    def __init__(self, single_particle_eigfunc, Nparticles):
        super(SlaterDetSampler, self).__init__()
        self.epsilon = 1e-5
        self.N = Nparticles         
        self.eigfunc = np.array(single_particle_eigfunc)
        self.D = self.eigfunc.shape[0]
        assert(self.N<=self.D)  
        # P-matrix representation of Slater determinant, (D x N)-matrix
        self.P = self.eigfunc[:,0:self.N]

        # U is the key matrix representing the Slater determinant for sampling purposes.
        # Its principal minors are the probabilities of certain particle configurations. 
        self.U = np.matmul(self.P, np.transpose(self.P))
        
        # for JASTROW FACTOR => should have its own class 
        self.v = nn.Parameter(torch.zeros(self.D, self.D).double() * 1e-1)
        with torch.no_grad(): # in-place operation on leaf tensor
            self.v[0, self.D-1] = self.v[self.D-1, 0] = 0.0
            for i in range(1,self.D):
                self.v[i-1, i] = self.v[i, i-1] = 0.0

        # Pauli_blocker imposes an ordering of the sampled positions:
        #  pos_0 < pos_1 < pos_2 < ... < pos_N. 
        # The particle sampled first should be on the leftmost position and so on.
        self.Pauli_blocker = torch.ones(self.D)

        self.reset_sampler()

    def reset_sampler(self):
        # Occupation numbers of the M sites up to the k-th sampling step
        # occ_vec[i]==1 for occupied and occ_vec[i]==0 for unoccupied i-th site.
        self.occ_vec = np.zeros(self.D, dtype=np.float64)
        self.occ_positions = torch.zeros(self.N, dtype=torch.int64).fill_(-10e6)
        # list of particle positions 
        self.Ksites = []
        
        # helper variables for low-rank update 
        # (These need to be updated after each sampling step.)
        # The matrix Xinv grows during the sampling.
        self.Xinv = np.zeros((1, 1), dtype=np.float64)
        
        # Correction term due to correlations between componets 
        # (correction to the probability for sampling each position, i.e. it is a vector)
        # for the current sampling step
        self.corr = np.zeros(self.D, dtype=np.float64)
                
        # State index of the sampler: no sampling step so far 
        self.state_index = -1
        
    def get_cond_prob_Slater(self, k):
        """
            Calculate the conditional probabilities in the k-th step of componentwise 
            direct sampling.
                        
            The precondition is that the sampler is in the state 
            after sampling (k-1) components. 
            
            Returns:
            --------
            cond_prob_i: arraylike
                pos_i \in {0,1,2,...,D-1} is the sampled position of the 
                k-th particle.                 
        """
        assert k < self.N 
        assert self.state_index == k-1, "state_index=%d, k=%d"%(self.state_index, k)  
        self.cond_prob = np.zeros(self.D, dtype=np.float64)
        
        if (k==0):
            # no correction term for the probability of the position of the first particle 
            self.cond_prob = np.diag(self.U)                    
        else:
            # correction term due to correlations between different particles 
            # (for each position on the lattice)
            self.corr = np.zeros(self.D, dtype=np.float64)
            for pos in np.arange(0, self.D):
                self.corr[pos] = np.matmul(
                      self.U[pos, self.Ksites[0:k]], 
                      np.matmul(self.Xinv[0:k, 0:k], self.U[self.Ksites[0:k], pos])
                    )        
            self.cond_prob[:] = np.diag(self.U) - self.corr[:]

        self.cond_prob_unnormalized = np.zeros(self.D, dtype=np.float64)    
        self.cond_prob_unnormalized[:] = self.cond_prob[:]
                
        # hack => IMPROVE AND UNDERSTAND
        self.cond_prob = self.cond_prob.real / (self.N - k)        
        
        # assert( np.all(self.cond_prob[:] >= 0) )
        #assert( (abs(np.sum(self.cond_prob[:])-1.0) < self.epsilon) )        
                
        return abs(self.cond_prob)

    def get_cond_prob_Jastrow(self, k):
        """
            Unnormalized relative probabilities 

            Precondition: k-1 particle positions have been sampled before. 
            
            The Jastrow factor is just a matrix of pairwise density-density interactions.
            Later, this should be replace by a neural network. 
        """
        if k > 0:
            probs = torch.exp(-0.5*torch.index_select(self.v, 1, self.occ_positions[0:k]).sum(dim=-1))
            return probs
        else:
            return torch.ones(self.D)

    def get_cond_prob(self, k):
        """
        Multiply conditional probabilities from Slater determinant and Jastrow factor
        and normalize (since Jastrow factor only gives unnormalized probabilities). 

        Enforce particle ordering. 
        """
        self.Pauli_blocker[:] = 1.0
        self.Pauli_blocker[self.D-self.N+k+1:] = 0.0
        if k > 0:
            # adjust Pauli blocker
            self.Pauli_blocker[:self.occ_positions[k-1]+1] = 0.0 

        probs_F = torch.tensor(self.get_cond_prob_Slater(k), requires_grad=False)
        probs_B = self.get_cond_prob_Jastrow(k)
        probs_ = probs_F * probs_B
        probs = probs_ / torch.sum(probs_)
        return probs


    def update_state(self, pos_i):
        """ 
            IMPROVE: pos_i should be batched 

            Having sampled position `pos_i` in the k-th step of componentwise direct sampling,
            update the state of the sampler so that it is ready to output the conditional 
            probability for the (k+1)-th step. 
            
            It is assumed that the conditional probability of the Slater determinant sampler is 
            combined with the weight coming from the Jastrow factor so that the actual sampling step
            happens in the calling routine. The result of this sampling step is fed back to the 
            Slater determinant sampler through the present `update_state` method.
            
            Input:
            ------
            pos_i: int \in {0, 1, 2, ..., D-1}
                Position of the particle sampled in the k-th step. 
        """        
        assert( 0 <= int(pos_i) < self.D )
        k = self.state_index + 1 
        self.Ksites += list([pos_i])  
            
        if k==0:
            self.Xinv = np.zeros((1,1), dtype=np.float64)
            self.Xinv[0,0] = 1.0 / self.U[self.Ksites[0], self.Ksites[0]]
        else:
            # low-rank update: 
            # Avoid computation of determinants and inverses by utilizing 
            # the formulae for determinants and iverse of block matrices. 
            # Compute Xinv based on the previous Xinv. 
            gg = 1.0 / self.cond_prob_unnormalized[pos_i]
            uu = np.matmul(self.Xinv, self.U[self.Ksites[0:k], self.Ksites[k]])
            vv = np.matmul(self.U[self.Ksites[k], self.Ksites[0:k]], self.Xinv)

            Xinv_new = np.zeros((k+1, k+1), dtype=np.float64)
            Xinv_new[0:k, 0:k] = self.Xinv[0:k, 0:k] + gg*np.outer(uu, vv)
            Xinv_new[k, 0:k] = -gg*vv[:]
            Xinv_new[0:k, k] = -gg*uu[:]
            Xinv_new[k,k] = gg 
            self.Xinv = Xinv_new 

        self.occ_vec[pos_i] = 1  
        self.occ_positions[k] = torch.tensor(pos_i, dtype=torch.int64)  # IMPROVE: keep torch and numpy separate (i.e. torch only in Jastrow factor)
        # Now the internal state of the sampler is fully updated.
        self.state_index += 1 
                
    def _sample_k(self,  k):
        """
            Sample position of the k-th particle via componentwise direct sampling. 
            
            The precondition is that the sampler is in the state 
            after sampling (k-1) components. The internal state of the sampler
            is updated before returning the sample for the k-th component. 
            
            Returns:
            --------
            pos: int 
                Position pos \in {0,1,...,D-1} of the k-th particle. 
        """
        assert k < self.N
        assert self.state_index == k-1
        
        # sample 
        probs = self.get_cond_prob(k) 

        pos = Categorical(probs).sample().numpy()
        
        # conditional prob in this sampling step
        cond_prob_k = probs[pos]

        # update state of the sampler given the result of the sampling step 
        self.update_state(pos)
        
        return pos, cond_prob_k 
        
    def chainevaluate_amplitude(self, config):
        """
        Follow the chain of conditional probabilities, skipping the sampling step and instead
        putting the actual position of the particle in `config`. This gives the probability
        (including both Slater determinant and Jastrow factor).
        To get the amplitude, take the square root and multiply the sign of the Slater determinant. 

        This is (probably) not more efficient than calculating the overlap of `config` with the 
        state |\psi> as a determinant. However, we need to go through the chain of conditional probabilities 
        and include the Jastrow factor in each step, because this is the way the Jastrow 
        factor is incorporated. (In Markov chain MC one could just multiply the Jastrow factor 
        to a configuration because only relative probabilities are needed and the normalization 
        is not required, but here we need normalized probabilities.)
        """   
        assert len(config.shape) == 1 # no batch dimension 
        config = np.array(config)
        print("config=", config)
        positions = np.nonzero(config)[0]
        print("chainevaluate, positions=", positions)
        self.reset_sampler()
        prob_sample = 1.0
        for k in np.arange(self.N):
            probs = self.get_cond_prob(k) # includes Jastrow factor
            pos = positions[k].item()
            cond_prob_k = probs[pos]
            print("pos=", pos, "cond_prob_k=", cond_prob_k)
            prob_sample *= cond_prob_k
            self.update_state(pos)

        return torch.sqrt(prob_sample) * torch.sign(torch.as_tensor(self.psi_amplitude(config)))


    def sample(self):
        """
            Sample a set of positions for N particles from the Slater determinant of 
            orthogonal orbitals. 

            Accumulate product of conditional probabilities. 
        """
        self.reset_sampler()
        
        prob_sample = 1.0
        for k in np.arange(self.N):
            _, cond_prob_k = self._sample_k(k)
            prob_sample *= cond_prob_k
            
        return self.occ_vec, prob_sample
    
    def psi_amplitude(self, samples):
        """
            Wavefunction amplitude for a basis state, i.e. the overlap
            between the occupation number state and the Slater determinant. 
                overlap = <i1, i2, ..., i_Np | \psi >
            
            Input:
            ------
                samples: binary array of occupation number states 
                    e.g. [[0,1,1,0],[1,1,0,0],[1,0,1,0]]

                    First dimension is batch dimension.
                    
            Returns:
            --------
                Overlap of the occupation number state `s` with the Slater 
                determinant. If the number of particles in `s` and the Slater 
                determinant is different, the overlap is mathematically zero. However, in this 
                case an assertion is violated rather than returning zero.

                First dimension is batch dimension.

            Example:  4 sites, 2 particles
            --------
            Illustrating how the batch dimension is silently taken care of by numpy / torch.
            >>> P = np.array([[0.2, 0.5],[0.25, 0.3],[0.25, 0.2], [0.3, 0.0]])
            >>> row_idx = np.array([[0,1],[2,3]]) # first dimension is batch dimension 
            >>> P[row_idx] # output should be a batch of 2x2 matrices
            array([[[0.2 , 0.5 ],
                    [0.25, 0.3 ]],
            <BLANKLINE>
                   [[0.25, 0.2 ],
                    [0.3 , 0.  ]]])
        """        
        row_idx = bin2pos(samples) 
        assert row_idx.shape[-1] == self.P.shape[-1]
        # select 2 (3,4,...) rows from a matrix with 2 (3,4,...) columns 
        submat = np.take(self.P, row_idx, axis=-2) # broadcast over leading dimensions of row_idx
        psi_amplitude = np.linalg.det(submat) / np.sqrt(self.N) #np.sqrt(math.factorial(self.N))
        return psi_amplitude.item()


    def psi_amplitude_ratio_Slater(self, r, s):
        pass
        
    def psi_amplitude_ratio_Jastrow(self, r, s):

        with torch.no_grad():
            pass


    def psi_amplitude_ratio(self, r, s):
        return self.psi_amplitude_ratio_Slater(r, s) * self.psi_amplitude_ratio_Jastrow(r, s)

    def psi_amplitude_I(self, samples_I):
        """
            Overlap of an occupation number state with the Slater determinant. 
            This is a wrapper function around `self.psi_amplitude(samples)`.
        """
        samples_I = np.array(samples_I)
        assert len(samples_I.shape) >= 1, "Input should be bitcoded integer (with at least one batch dim.)."
        samples = int2bin(samples_I, self.D)
        return self.psi_amplitude(samples)  


    def log_prob(self, samples):
        """
            Logarithm of the modulus squared of the wave function in a basis state.
                    2 * log ( | < i1, i2, ..., i_Np | \psi > | )     
        """
        return 2 * np.log(np.abs(self.psi_amplitude(samples)))

    def log_prob_I(self, samples_I):
        return 2 * np.log(np.abs(self.psi_amplitude_I(samples_I)))


def _test():
    import doctest
    doctest.testmod(verbose=True)


def code_verification():
    import matplotlib.pyplot as plt 
    from time import time 
    from test_suite import ( 
        prepare_test_system_zeroT,
        Slater2spOBDM
    )

    (Nsites, eigvecs) = prepare_test_system_zeroT(Nsites=200, potential='none', PBC=False)

    for Np in (30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,):
        Nparticles = Np
        num_samples = 5
        print("Nsites=", Nsites, "Nparticles=", Nparticles, "num_samples=", num_samples)
        SDsampler = SlaterDetSampler(eigvecs, Nparticles=Nparticles)

        # Check that sampling the Slater determinant gives the correct average density. 
        occ_vec = torch.zeros(Nsites)
        t0 = time()
        for s in range(num_samples):
            occ_vec_, prob_sample = SDsampler.sample()
            #occ_vec += occ_vec_
        t1 = time()
        print("elapsed, unordered sampling=", (t1-t0))
        
    ##print("occ_vec=", occ_vec)    
    #density = occ_vec / float(num_samples)
    ##print("density=", density)

    #OBDM = Slater2spOBDM(eigvecs[:, 0:Nparticles])

    #f = plt.figure(figsize=(8,6))
    #ax = f.add_subplot(1,1,1)
    #ax.set_xlabel(r"site $i$")
    #ax.set_ylabel(r"av. density")
    #ax.plot(range(len(density)), density, label=r"av.density $\langle n \rangle$ (sampled)")
    #ax.plot(range(len(np.diag(OBDM))), np.diag(OBDM), label=r"$\langle n \rangle$ (from OBDM)")
    #plt.legend()
    ##plt.show()


if __name__ == '__main__':

    from utils import default_dtype_torch
    torch.set_default_dtype(default_dtype_torch)
    
    #_test()

    code_verification()
