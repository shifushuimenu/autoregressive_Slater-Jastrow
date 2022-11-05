# TODO:
#  - longrange 2D, lowrank update: distinguish cases where detG_num is singular
#    (like in the n.n. case)
#  - There are additional cases for longrange 2D:
#         (1) i > (r,s).
#         (2) r and s are both occupied by the first particle. 
#  - _detratio_from_scratch() should only be called when an error ErrorFinitePrecision is thrown.
#    Other exceptional cases should have a customized solution (using a lowrank update).
#  - Make sure reference_state_2D.py is synchronized after the above corrections have been made. 
#  - Raise an error ErrorFinitePrecision wherever this can occur ( also in lowrank update for n.n. hopping)
#  - Document each and every type of lowrank update (DONE)
#  - In critical points scipy.linalg is used instead of np.linalg as the former supports float128. 
#  - Replace np.isclose() by something faster.
#
#  - Is it possible to avoid 
#            Gdenom_inv = np.linalg.inv(Gdenom)
#    in lowrank_kinetic() by reusing information of state alpha from the sampling step ?
#  - Standardize interface for slater_sampler 

import torch
import numpy as np
import math 

from torch.distributions.categorical import Categorical
from bitcoding import bin2pos, int2bin

from time import time 
from lowrank_update_kinetic import lowrank_update_kinetic 

#from profilehooks import profile
import matplotlib.pyplot as plt 

class SlaterDetSampler_ordered(torch.nn.Module):
    """
        Sample a set of particle positions from a Slater determinant of 
        orthogonal orbitals via direct componentwise sampling, thereby
        making sure that the particle positions come out ordered.
                
        Parameters:
        -----------
        single_particle_eigfunc: 2D arraylike or None
            Matrix of dimension D x D containing all the single-particle
            eigenfunctions as columns. Note that D is the dimension
            of the single-particle Hilbert space.
        Nparticles: int
            Number of particles where `Nparticles` <= `D`.

        From the matrix containing the single-particle eigenfunctions 
        as columns, the first `Nparticles` columns are chosen to form the 

        Slater determinant.
    """
    def __init__(self, Nsites, Nparticles, single_particle_eigfunc=None, eigvals=None, naive_update=True, optimize_orbitals=False, outdir=None):
        super(SlaterDetSampler_ordered, self).__init__()
        self.epsilon = 1e-5
        self.D = Nsites 
        self.N = Nparticles         
        assert(self.N<=self.D)  
        self.naive_update = naive_update
        # co-optimize also the columns of the Slater determinant
        self.optimize_orbitals = optimize_orbitals
        self.dir = outdir if outdir is not None else "./"

        self.t_det_Schur_complement = 0.0
        self.t_npix_ = 0.0
        self.t_get_cond_prob = 0.0
        self.t_update_state = 0.0
        self.t_lowrank_linalg = 0.0
        self.t_linstorage = 0.0
        self.t_gemm = 0.0

        if single_particle_eigfunc is not None: 
           self.eigfunc = np.array(single_particle_eigfunc)
           assert Nsites == self.eigfunc.shape[0]
        else: # random initialization of orbitals  
           self.optimize_orbitals = True 
           # generate random unitary 
           T0 = torch.tril(torch.rand(self.N, self.N, requires_grad=False), diag=0)
           self.eigfunc = torch.matrix_exp(T0 - T0.t())

        # P-matrix representation of Slater determinant, (D x N)-matrix
        self.P = torch.tensor(self.eigfunc[:,0:self.N], requires_grad=False)

        if self.optimize_orbitals:
            # parametrized rotation matrix R for optimizing the orbitals by orbital rotation
            self.T = torch.nn.Parameter(torch.zeros(self.D, self.D)) # leaf Variable, updated during SGD
            T_ = torch.tril(self.T, diagonal=-1) # only lower triangular elements are relevant 
            self.R = torch.matrix_exp(T_ - T_.t())        
            self.P_ortho = self.R @ self.P
            print("is leaf ? self.T=", self.T.is_leaf)
        else:
            self.P_ortho = self.P
 
        # called also inside self.reset_sampler() if self.optimize_orbitals == True 
        self.rotate_orbitals()

        self.reset_sampler()

    def rotate_orbitals(self):
        if self.optimize_orbitals:
            T_ = torch.tril(self.T, diagonal=-1) # only lower triangular elements are relevant 
            self.R = torch.matrix_exp(T_ - T_.t())
            self.P_ortho = self.R @ self.P
            # U is the key matrix representing the Slater determinant for *unordered* sampling purposes.
            # Its principal minors are the probabilities of certain particle configurations.
            self.U = self.P_ortho @ self.P_ortho.t()
            # The Green's function is the key matrix for *ordered* sampling. 
            self.G = torch.eye(self.D) - self.U    
        else:
            self.U = self.P_ortho @ self.P_ortho.t()
            self.G = torch.eye(self.D) - self.U   

    def reset_sampler(self, rebuild_comp_graph=True):        
        self.occ_vec = np.zeros(self.D, dtype=np.float64)
        self.occ_positions = np.zeros(self.N, dtype=np.int64)
        self.occ_positions[:] = -10^6 # set to invalid values 
        # list of particle positions 
        self.Ksites = []
        self.xmin = 0
        self.xmax = self.D - self.N + 1

        # State index of the sampler: no sampling step so far 
        self.state_index = -1

        # To avoid backward(retain_graph=True) when backpropagating on psi_loc for each config
        # the computational subgraph involving the Slater determinant needs to be (unnecessarily)
        # recomputed because of the dynamical computation graph of pytorch. 
        # This is only necessary during density estimation. 
        if self.optimize_orbitals and rebuild_comp_graph:
            self.rotate_orbitals()



    #@profile
    def get_cond_prob(self, k):
        r""" Conditional probability for the position x of the k-th particle.

             The support x \in [xmin, xmax-1] of the distribution changes with `k`.
        """
        t_tot1=time()
        self.xmin = 0 if k==0 else self.occ_positions[k-1] + 1 
        self.xmax = self.D - self.N + k + 1

        probs = torch.zeros(self.D) #np.zeros(len(range(self.xmin, self.xmax)))

        if self.naive_update:
            Ksites_tmp = self.Ksites[:]
            occ_vec_tmp = self.occ_vec[:]
            ### Not needed anymore when cancelling the denominator 
            GG_denom = self.G[np.ix_(self.Ksites, self.Ksites)] - torch.diag(torch.tensor(self.occ_vec[0:len(self.Ksites)]))
        else:
            self.BB_reuse = []   # list of matrices to be reused later 
            self.Schur_complement_reuse = []  # list of matrices to be reused later 


        ## alternative way of constructing BB and DD matrices 
        ## This imposes an ordering, i.e. the Ksites-array does no longer abstract from 
        ## the ordering of sites in the Green's function !
        t1 = time()
        # Pre-fetch from G the full possible range of sites for DD and BB block. 
        if self.Ksites == []:
            self.BB_linstorage = np.array([])
        else:
            self.BB_linstorage = self.G[self.Ksites[0]:self.Ksites[-1]+1, self.xmin:self.xmax].T.flatten() # careful: This is a view into G.
        self.DD_full = self.G[self.xmin:self.xmax, self.xmin:self.xmax] # careful: This is a view into G.
        t2 = time()
        self.t_linstorage += (t2-t1)


        mm=-1
        for i_k in range(self.xmin, self.xmax):
            mm += 1
            if self.naive_update:
                Ksites_tmp.append(i_k)
                occ_vec_tmp[i_k] = 1           
            
                GG_num = self.G[np.ix_(Ksites_tmp, Ksites_tmp)] - torch.diag(torch.tensor(occ_vec_tmp[0:i_k+1]))                
                probs[i_k] = (-1) * torch.det(GG_num) / torch.det(GG_denom)
                                
                # Cancel numerator determinant rather than calculating it. 
                if k > 0:
                    base_pos_ = 0 if i_k==0 else self.Ksites[-1]    
                    tmp = (-1) * self._detratio_from_scratch(self.G.numpy(), occ_vec=occ_vec_tmp, base_pos=base_pos_, i=i_k)
                else: #k == 0
                    GG_num = self.G[np.ix_(Ksites_tmp, Ksites_tmp)] - torch.diag(torch.tensor(occ_vec_tmp[0:i_k+1]))
                    tmp = (-1) * torch.det(GG_num)
                #print("k=", k, "??? tmp=", tmp, "probs[i_k]=", probs[i_k])
                #assert torch.isclose(torch.tensor([tmp]), probs[i_k])
                probs[i_k] = torch.tensor([tmp])

                occ_vec_tmp[i_k] = 0  # reset for next loop iteration

            else: # use block determinant formula
                #Ksites_add = list(range(self.xmin, i_k+1))
                #occ_vec_add = torch.tensor([0] * (i_k - self.xmin) + [1])
                #t1 = time()
                #NN = torch.diag(occ_vec_add[:])
                #DD = self.G[np.ix_(Ksites_add, Ksites_add)] - NN
                #self.BB = self.G[np.ix_(self.Ksites, Ksites_add)]
                #t2 = time()
                #self.t_npix_ += (t2-t1)


                ###alternative approach of constructing BB and DD matrices
                # Use large pre-fetched DD and BB blocks to construct the current ones 
                assert mm == i_k - self.xmin
                ll = mm + 1 # mm = i_k - self.xmin
                t1 = time()
                self.BB = self.BB_linstorage[0:ll*self.xmin].reshape(ll, self.xmin).T
                occ_vec_add = torch.tensor([0] * mm + [1])
                NN = torch.diag(occ_vec_add[:])
                DD = self.DD_full[0:mm+1, 0:mm+1] - NN
                t2 = time()
                self.t_linstorage += (t2-t1)
                #assert np.isclose(self.BB, self.BB_v2).all()
                #assert np.isclose(DD, DD_v2).all()
                ## end alternative approach 


                if len(self.BB) == 0:
                   CC = self.BB
                else:
                   CC = self.BB.transpose(-1, -2)
                self.BB_reuse.append(self.BB)
                if self.state_index==-1: # no sampling step so far 
                    # here self.Xinv = [] always. IMPROVE: This line is useless.
                    #self.Xinv = torch.linalg.inv(self.G[np.ix_(self.Ksites, self.Ksites)] - torch.diag(torch.tensor(self.occ_vec[0:self.xmin])))
                    self.Xinv = torch.tensor([])
                    #print("CALLED?, self.Xinv=", self.Xinv)
                else:
                    pass # self.Xinv should have been updated by calling update_state(pos_i)
                if len(self.BB) == 0:
                   self.Schur_complement = DD - torch.tensor([0.0])
                else:
                   t1 = time()
                   self.Schur_complement = DD - torch.matmul(torch.matmul(CC, self.Xinv), self.BB)
                   t2 = time()
                   self.t_gemm += (t2-t1)

                   #print("dimensions: CC.size()=", CC.size()[0], "Xinv.size()=", self.Xinv.size()[0], "self.BB.size()=", self.BB.size()[0], "Schur=", self.Schur_complement.size()[0])
                self.Schur_complement_reuse.append(self.Schur_complement)
                # for small matrix sizes (1,2,3), the determinant should be "hand-coded" for speed-up
                #print("Schur complement size=", self.Schur_complement.size()[0])
                t1 = time()
                probs[i_k] = (-1) * torch.det(self.Schur_complement)
                t2 = time()
                self.t_det_Schur_complement += (t2-t1)
                #print("Schur_complement.shape=", self.Schur_complement.shape, "det, elapsed=", t2-t1)

        if not self.naive_update:
            assert len(self.BB_reuse) == len(self.Schur_complement_reuse) == (mm+1) # IMPROVE: remove this as well as the variable mm
        # IMPROVE: For large matrices the ratio of determinants leads to numerical
        # instabilities which results in not normalized probability distributions   
        # => use LOW-RANK UPDATE     
        #print("k=", k, "xmin=", self.xmin, "xmax=", self.xmax, "i_k=", i_k, "probs[:]=", probs, "  np.sum(probs[:])=", probs.sum())         
        #np.savetxt("Schur_complement.dat", self.Schur_complement)
        assert torch.isclose(probs.sum(), torch.tensor([1.0])), "k=%d, norm=%20.16f" % (k, probs.sum()) # assert normalization 
        # clamp negative values which are in absolute magnitude below machine precision
        probs = torch.where(abs(probs) > 1e-15, probs, torch.tensor([0.0]))

        assert not torch.any(torch.isnan(probs))
        t_tot2=time()
        self.t_get_cond_prob += (t_tot2 - t_tot1)

        return probs 

    def _sample_k(self, k):
        assert k < self.N 
        assert self.state_index == k-1

        probs = self.get_cond_prob(k)
        pos = Categorical(torch.Tensor(probs)).sample().numpy()
        # conditional prob in this sampling step 
        cond_prob_k = probs[pos]

        # update state of the sampler given the result of the sampling step 
        self.update_state(pos.item())

        return pos, cond_prob_k

    def sample(self):
        self.reset_sampler()

        prob_sample = 1.0
        for k in np.arange(0, self.N):
            _, cond_prob_k = self._sample_k(k)
            prob_sample *= cond_prob_k
        
        return self.occ_vec, prob_sample

    #@profile
    def update_state(self, pos_i):
        t_tot1 = time()
        assert type(pos_i) == int 
        assert( 0 <= pos_i < self.D )
        k = self.state_index + 1

        self.Ksites.extend(list(range(self.xmin, pos_i+1)))
        self.occ_vec[self.xmin:pos_i] = 0
        self.occ_vec[pos_i] = 1
        self.occ_positions[k] = pos_i
        if not self.naive_update:
            # Update Xinv based on previous Xinv using 
            # formula for inverse of a block matrix 
            if self.state_index == -1: # first update step 
                Ksites_add = list(range(0, pos_i+1))
                occ_vec_add = [0] * pos_i + [1]
                if len(occ_vec_add) > 1: # np.diag() does not work for one-element arrays 
                    NN = torch.diag(torch.tensor(occ_vec_add[:]))
                else:
                    NN = torch.tensor(occ_vec_add)
                self.Xinv_new = torch.linalg.inv(self.G[np.ix_(Ksites_add, Ksites_add)] - NN)
                self.Xinv = self.Xinv_new
            else:                
                Ksites_add = list(range(self.xmin, pos_i+1))  # put the sampled pos_i instead of loop variable i_k
                occ_vec_add = [0] * (pos_i - self.xmin) + [1] # IMPROVE: pos_i is a tensor here, which is not necessary 
                NN = torch.diag(torch.tensor(occ_vec_add[:]))

                mm = pos_i - self.xmin  # the m-th position on the current support of the conditional probs
                BB_ = self.BB_reuse[mm]      # select previously computed matrices for the sampled position (i.e. pos_i, which is the m-th position of the current support)
                Schur_complement_ = self.Schur_complement_reuse[mm]

                XinvB = torch.matmul(self.Xinv, BB_) 
                t1 = time()
                Sinv = torch.linalg.inv(Schur_complement_)
                t2 = time()
                #remove
                #print("Schur_complement_.shape=", Schur_complement_.shape, "linalg.inv, elapsed=", t2-t1)
                #remove
                Ablock = (self.Xinv + 
                    torch.matmul(torch.matmul(XinvB, Sinv), XinvB.transpose(-1,-2)))
                Bblock = - torch.matmul(XinvB, Sinv)
                Cblock = - torch.matmul(Sinv, XinvB.transpose(-1,-2))
                Dblock = Sinv 
                self.Xinv_new = torch.vstack(( torch.hstack((Ablock, Bblock)), torch.hstack((Cblock, Dblock)) ))  # np.block([[Ablock, Bblock], [Cblock, Dblock]])
                self.Xinv = self.Xinv_new

        self.state_index += 1 
        t_tot2 = time()
        self.t_update_state += (t_tot2 - t_tot1)

    #@profile
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
            NOT VALID ANYMORE 
            >>> P = np.array([[0.2, 0.5],[0.25, 0.3],[0.25, 0.2], [0.3, 0.0]])
            >>> row_idx = np.array([[0,1],[2,3]]) # first dimension is batch dimension 
            >>> P[row_idx] # output should be a batch of 2x2 matrices
            array([[[0.2 , 0.5 ],
                    [0.25, 0.3 ]],
            <BLANKLINE>
                   [[0.25, 0.2 ],
                    [0.3 , 0.  ]]])
        """                
        row_idx = torch.Tensor(bin2pos(samples)).to(torch.long) # tenGdenom_inv = np.linalg.inv(Gdenom)sors used as indices must be long
        assert row_idx.shape[-1] == self.P_ortho.shape[-1]
        # select 2 (3,4,...) rows from a matrix with 2 (3,4,...) columns 
        # submat = np.take(self.P, row_idx, axis=-2) # broadcast over leading dimensions of row_idx
        submat = self.P_ortho[..., row_idx, :] # row_idx[0] -> UNDO BATCH DIMENSION, but the determinant is correct even with a first batch dimension
        psi_amplitude = torch.det(submat)
        return psi_amplitude 


    def psi_amplitude_I(self, samples_I):
        """
            Overlap of an occupation number state with the Slater determinant. 
            This is a wrapper function around `self.psi_amplitude(samples)`.

            Input:
            ------
                samples_I: integer representation I of a binary array.
                    First dimension is batch dimension. 
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
        
    
    def lowrank_kinetic(self, ref_I, xs_I, rs_pos, print_stats=True):
        """
            Calculate local kinetic energy in state alpha via lowrank 
            update of conditional probabilities of the reference state alpha.
        """
        GG = self.G.detach().numpy()
        return lowrank_update_kinetic(GG, self.D, self.N, ref_I, xs_I, rs_pos, print_stats)



def _test():
    import doctest 
    doctest.testmod(verbose=True)


if __name__ == "__main__":
    
    _test()
