import torch
import numpy as np

from torch.distributions.categorical import Categorical
from bitcoding import bin2pos, int2bin


class SlaterDetSampler_ordered(torch.nn.Module):
    """
        Sample a set of particle positions from a Slater determinant of 
        orthogonal orbitals via direct componentwise sampling, thereby
        making sure that the particle positions come out ordered.
                
        Parameters:
        -----------
        single_particle_eigfunc: 2D arraylike
            Matrix of dimension D x D containing all the single-particle
            eigenfunctions as columns. Note that D is the dimension
            of the single-particle Hilbert space.
        Nparticles: int
            Number of particles where `Nparticles` <= `D`.

        From the matrix containing the single-particle eigenfunctions 
        as columns, the first `Nparticles` columns are chosen to form the 
        Slater determinant.
    """
    def __init__(self, single_particle_eigfunc, Nparticles, naive=True):
        super(SlaterDetSampler_ordered, self).__init__()
        self.epsilon = 1e-5
        self.N = Nparticles         
        self.eigfunc = np.array(single_particle_eigfunc)
        self.D = self.eigfunc.shape[0]
        assert(self.N<=self.D)  
        # P-matrix representation of Slater determinant, (D x N)-matrix
        self.P = self.eigfunc[:,0:self.N]

        self.naive_update = naive

        # U is the key matrix representing the Slater determinant for sampling purposes.
        # Its principal minors are the probabilities of certain particle configurations. 
        self.U = np.matmul(self.P, np.transpose(self.P))

        # Green's function 
        self.G = np.eye(self.D) - self.U

        self.reset_sampler()

    def reset_sampler(self):        
        self.occ_vec = np.zeros(self.D, dtype=np.float64)
        self.occ_positions = np.zeros(self.N, dtype=np.int64)
        self.occ_positions[:] = -10^6 # set to invalid values 
        # list of particle positions 
        self.Ksites = []
        self.xmin = 0
        self.xmax = self.D - self.N + 1

        # State index of the sampler: no sampling step so far 
        self.state_index = -1

        # helper variables for low-rank update


    def get_cond_prob(self, k):
        r""" Conditional probability for the position x of the k-th particle.

             The support x \in [xmin, xmax-1] of the distribution changes with `k`.
        """
        self.xmin = 0 if k==0 else self.occ_positions[k-1] + 1 
        self.xmax = self.D - self.N + k + 1

        probs = np.zeros(self.D) #np.zeros(len(range(self.xmin, self.xmax)))

        if self.naive_update:
            Ksites_tmp = self.Ksites[:]
            occ_vec_tmp = self.occ_vec[:]
            GG_denom = self.G[np.ix_(self.Ksites, self.Ksites)] - np.diag(self.occ_vec[0:len(self.Ksites)])

        for i_k in range(self.xmin, self.xmax):
        
            if self.naive_update:
                Ksites_tmp.append(i_k)
                occ_vec_tmp[i_k] = 1
                GG_num = self.G[np.ix_(Ksites_tmp, Ksites_tmp)] - np.diag(occ_vec_tmp[0:i_k+1])
                occ_vec_tmp[i_k] = 0  # reset for next loop
                probs[i_k] = - np.linalg.det(GG_num) / np.linalg.det(GG_denom)
            else: # use block determinant formula
                self.Ksites_old = self.Ksites.copy()
                Ksites_add = list(range(self.xmin, i_k+1))
                occ_vec_add = [0] * (i_k - self.xmin) + [1]
                if len(occ_vec_add) > 1:
                    NN = np.diag(occ_vec_add[:])
                else:
                    NN = occ_vec_add
                self.DD = self.G[np.ix_(Ksites_add, Ksites_add)] - NN
                self.CC = self.G[np.ix_(Ksites_add, self.Ksites_old)]
                self.BB = self.G[np.ix_(self.Ksites_old, Ksites_add)]
                if self.state_index==-1: # no sampling step so far 
                    # here self.Xinv = [] always. IMPROVE: This line is useless.
                    self.Xinv = np.linalg.inv(self.G[np.ix_(self.Ksites_old, self.Ksites_old)] - np.diag(self.occ_vec[0:self.xmin]))
                else:
                    pass # self.Xinv should have been updated by calling update_state(pos_i)
                self.Schur_complement = self.DD - np.matmul(np.matmul(self.CC, self.Xinv), self.BB)
                probs[i_k] = (-1) * np.linalg.det(self.Schur_complement)

        # IMPROVE: For large matrices the ratio of determinants leads to numerical
        # instabilities which results in not normalized probability distributions   
        # => use LOW-RANK UPDATE     
        #print("probs[:]=", probs[:], "  np.sum(probs[:])=", np.sum(probs[:])) 
        assert np.isclose(np.sum(probs[:]), 1.0) # assert normalization 
        # clamp negative values which are in absolute magnitude below machine precision
        probs = np.where(abs(probs) > 1e-15, probs, 0.0)

        return probs 

    def _sample_k(self, k):
        assert k < self.N 
        assert self.state_index == k-1

        probs = self.get_cond_prob(k)
        pos = Categorical(torch.Tensor(probs)).sample().numpy()
        # conditional prob in this sampling step 
        cond_prob_k = probs[pos]

        # update state of the sampler given the result of the sampling step 
        self.update_state(pos)

        return pos, cond_prob_k

    def sample(self):
        self.reset_sampler()

        prob_sample = 1.0
        for k in np.arange(0, self.N):
            _, cond_prob_k = self._sample_k(k)
            prob_sample *= cond_prob_k
        
        return self.occ_vec, prob_sample


    def update_state(self, pos_i):

        assert( 0 <= int(pos_i) < self.D )
        k = self.state_index + 1

        self.Ksites.extend(list(range(self.xmin, pos_i+1)))
        self.occ_vec[self.xmin:pos_i] = 0
        self.occ_vec[pos_i] = 1
        self.occ_positions[k] = pos_i

        if not self.naive_update:
            # preprare helper variables for the next pass 
            # update Xinv based on previous Xinv

            if self.state_index == -1: # first update step 
                Ksites_add = list(range(0, pos_i+1))
                occ_vec_add = [0] * pos_i + [1]
                if len(occ_vec_add) > 1: # np.diag() does not work for one-element arrays 
                    NN = np.diag(occ_vec_add[:])
                else:
                    NN = occ_vec_add
                self.Xinv_new = np.linalg.inv(self.G[np.ix_(Ksites_add, Ksites_add)] - NN)
                self.Xinv = self.Xinv_new
            else:                
                Ksites_add = list(range(self.xmin, pos_i+1))  # put the sampled pos_i instead of loop variable i_k
                occ_vec_add = [0] * (int(pos_i) - self.xmin) + [1] # IMPROVE: pos_i is a tensor here, which is not necessary 
                if len(occ_vec_add) > 1:
                    NN = np.diag(occ_vec_add[:])
                else:
                    NN = occ_vec_add
                self.DD = self.G[np.ix_(Ksites_add, Ksites_add)] - NN
                self.CC = self.G[np.ix_(Ksites_add, self.Ksites_old)]
                self.BB = self.G[np.ix_(self.Ksites_old, Ksites_add)]
                self.Schur_complement = self.DD - np.matmul(np.matmul(self.CC, self.Xinv), self.BB)

                XinvB = np.matmul(self.Xinv, self.BB) 
                Sinv = np.linalg.inv(self.Schur_complement)
                Ablock = (self.Xinv + 
                    np.matmul(np.matmul(XinvB, Sinv), XinvB.transpose()))
                Bblock = - np.matmul(XinvB, Sinv)
                Cblock = - np.matmul(Sinv, XinvB.transpose())
                Dblock = Sinv 
                #self.Xinv_new = np.vstack((np.hstack((Ablock, Bblock)), np.hstack((Cblock, Dblock))))
                self.Xinv_new = np.block([[Ablock, Bblock], [Cblock, Dblock]])
                self.Xinv = self.Xinv_new

        self.state_index += 1 


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
        row_idx = torch.Tensor(bin2pos(samples)).to(torch.long) # tensors used as indices must be long
        assert row_idx.shape[-1] == self.P.shape[-1]
        # select 2 (3,4,...) rows from a matrix with 2 (3,4,...) columns 
        submat = np.take(self.P, row_idx, axis=-2) # broadcast over leading dimensions of row_idx
        psi_amplitude = np.linalg.det(submat)
        return psi_amplitude 


    def psi_amplitude_I(self, samples_I):
        """
            Overlap of an occupation number state with the Slater determinant. 

            This is a wrapper function around `self.psi_amplitude(samples)`.

            Input:
            ------
                samples_I: integer representation I of a binary array.
                    First dimension is batch dimension. 

            Returns:
            --------

            Example:
            --------
        """
        samples_I = torch.as_tensor(samples_I)
        assert len(samples_I.shape) >= 1, "Input should be bitcoded integer (with at least one batch dim.)."
        samples = int2bin(samples_I, self.D)
        return self.psi_amplitude(samples)  

    def log_prob(self, samples):
        """
            Logarithm of the modulus squared of the wave function in a basis state.
                    2 * log ( | < i1, i2, ..., i_Np | \psi > | )     
        """
        return 2 * torch.log(torch.abs(self.psi_amplitude(samples)))


if __name__ == "__main__":

    import matplotlib.pyplot as plt 
    from test_suite import ( 
        prepare_test_system_zeroT,
        Slater2spOBDM
    )

    (Nsites, eigvecs) = prepare_test_system_zeroT(Nsites=10, potential='none', PBC=False)
    Nparticles = 3
    num_samples = 40000

    SDsampler = SlaterDetSampler_ordered(eigvecs, Nparticles=Nparticles)

    # Check that sampling the Slater determinant gives the correct average density. 
    occ_vec = torch.zeros(Nsites)
    for s in range(num_samples):
        occ_vec_, prob_sample = SDsampler.sample()
        occ_vec += occ_vec_
        
    print("occ_vec=", occ_vec)    
    density = occ_vec / float(num_samples)
    print("density=", density)

    OBDM = Slater2spOBDM(eigvecs[:, 0:Nparticles])

    f = plt.figure(figsize=(8,6))
    ax = f.add_subplot(1,1,1)
    ax.set_xlabel(r"site $i$")
    ax.set_ylabel(r"av. density")
    ax.plot(range(len(density)), density, label=r"av.density $\langle n \rangle$ (sampled)")
    ax.plot(range(len(np.diag(OBDM))), np.diag(OBDM), label=r"$\langle n \rangle$ (from OBDM)")
    plt.legend()
    plt.show()
