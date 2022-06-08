# TODO: - replace hstack and vstack by directly writing to preallocated memory 
#         For this the sizes of the blocks need to be known explicitly. (DONE)
#       - Rather than calculating the determinant of the Schur complement use the formula
#         for the determinant of a block matrix. (DONE)
#       - Is the Schur complement a symmetric matrix ? Is yes, then use this to reduce memory access. 
#
#       - Check whether conditional probabilities are saturated and stop calculating cond. probs. for 
#         further positions. 

import torch
import torch.nn as nn
import numpy as np
from utils import default_dtype_torch

from torch.distributions.categorical import Categorical
from bitcoding import bin2pos, int2bin
from memory_layout import store_G_linearly, idx_linearly_stored_G, \
                          idx_linearly_stored_G_blockB1

from block_update import block_update_inverse 

#from profilehooks import profile
from time import time 


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
    def __init__(self, Nsites, Nparticles, single_particle_eigfunc=None, naive_update=True, eps_norm_probs=None):
        super(SlaterDetSampler_ordered, self).__init__()
        self.epsilon = 1e-5
        self.eps_norm_probs = 1.0 - 1e-6 if eps_norm_probs is None else eps_norm_probs
        self.D = Nsites 
        self.N = Nparticles         
        assert(self.N<=self.D)  
        self.naive_update = naive_update
        if single_particle_eigfunc is not None: 
           self.optimize_orbitals = False 
           self.eigfunc = np.array(single_particle_eigfunc)
           assert Nsites == self.eigfunc.shape[0]
           # P-matrix representation of Slater determinant, (D x N)-matrix
           self.P = torch.tensor(self.eigfunc[:,0:self.N])
           self.P_ortho = self.P 
           # U is the key matrix representing the Slater determinant for sampling purposes.
           # Its principal minors are the probabilities of certain particle configurations.            
           self.U = torch.matmul(self.P, self.P.transpose(-1,-2)) 
           # Green's function 
           self.G = torch.eye(self.D) - self.U
           # efficient memory layout optimized for memory access pattern of algorithm 
           self.G_lin_mem = torch.tensor(store_G_linearly(self.G.numpy())) # IMPROVE: sort out torch vs np  
       
        else: # optimize also the columns of the Slater determinant 
           self.optimize_orbitals = True 
           self.P = nn.Parameter(torch.rand(self.D, self.N, requires_grad=True)) # leaf Variable, updated during SGD; columns are not (!) orthonormal 
           self.reortho_orbitals()  # orthonormalize columns 

        self.cond_max = 0.0
        # timing 
        self.t_fetch_memory = 0.0
        self.t_matmul = 0.0
        self.t_update_Schur = 0.0
        self.t_det = 0.0

        print("requires grad ?:", self.U.requires_grad)
        print("self.P_ortho.is_leaf =", self.P_ortho.is_leaf)
        print("self.P.is_leaf =", self.P.is_leaf)

        self.reset_sampler()

    def reortho_orbitals(self):
        if self.optimize_orbitals:
           with torch.no_grad():
              self.P_ortho, R = torch.linalg.qr(self.P, mode='reduced') # P-matrix with orthonormal columns; on the other hand, it is self.P which is updated during SGD.

           self.P = nn.Parameter(self.P_ortho.detach())
           self.U = torch.matmul(self.P, self.P.transpose(-1,-2))
           # Green's function 
           self.G = torch.eye(self.D) - self.U  
           self.G_lin_mem = torch.tensor(store_G_linearly(self.G.numpy())) # IMPROVE: sort out torch vs np
        else:
           pass 

    def reset_sampler(self):        
        self.occ_vec = np.zeros(self.D, dtype=np.float64)
        self.occ_positions = np.zeros(self.N, dtype=np.int64)
        self.occ_positions[:] = -10^6 # set to invalid values 
        self.cond_probs = np.zeros(self.N*self.D) # cond. probs. for all components (for monitoring purposes)
        # list of particle positions 
        self.Ksites = []
        self.len_Ksites = 0
        self.xmin = 0
        self.xmax = self.D - self.N + 1

        # State index of the sampler: no sampling step so far 
        self.state_index = -1

        # helper variables for low-rank update


    #@profile
    def get_cond_prob(self, k):
        r""" Conditional probability for the position x of the k-th particle.

             The support x \in [xmin, xmax-1] of the distribution changes with `k`.
        """
        self.xmin = 0 if k==0 else self.occ_positions[k-1] + 1 
        self.xmax = self.D - self.N + k + 1

        probs = torch.zeros(self.D) #np.zeros(len(range(self.xmin, self.xmax)))
        cumul_probs = 0.0

        if self.naive_update:
            Ksites_tmp = self.Ksites[:]
            occ_vec_tmp = self.occ_vec[:]
            GG_denom = self.G[np.ix_(self.Ksites, self.Ksites)] - torch.diag(torch.tensor(self.occ_vec[0:len(self.Ksites)]))
        else:
            self.BB_reuse = []   # list of matrices to be reused later 
            self.Schur_complement_reuse = []  # list of matrices to be reused later 
            self.CCXinvBB_reuse = [] # almost the same as Schur complement, except for D block   
            # preallocate larger memory suitable for all iteration steps 
            CCXinvBB = torch.empty(self.xmax-self.xmin, self.xmax-self.xmin, dtype=default_dtype_torch) 
            DDsubmat = torch.empty(self.xmax-self.xmin, self.xmax-self.xmin, dtype=default_dtype_torch)
            BtXinv_new = torch.empty(self.xmax-self.xmin, self.len_Ksites, dtype=default_dtype_torch)
            BBsubmat = torch.empty(self.len_Ksites, self.xmax-self.xmin, dtype=default_dtype_torch)

        mm=-1
        for i_k in range(self.xmin, self.xmax):

            mm += 1
            if self.naive_update:
                Ksites_tmp.append(i_k)
                occ_vec_tmp[i_k] = 1

                if mm == 0:
                    Gsubmat = self.G[np.ix_(Ksites_tmp, Ksites_tmp)]
                else:
                    A = Gsubmat 
                    B = self.G[np.ix_(Ksites_tmp[:-1], [Ksites_tmp[-1]])]
                    C = self.G[np.ix_([Ksites_tmp[-1]], Ksites_tmp[:-1])]
                    D = self.G[Ksites_tmp[-1], Ksites_tmp[-1]][None, None]  # convert tensor element to 2d array 
                    Gsubmat = torch.vstack((torch.hstack((A, B)), torch.hstack((C, D))))    

                GG_num = Gsubmat - torch.diag(torch.tensor(occ_vec_tmp[0:i_k+1]))
                occ_vec_tmp[i_k] = 0  # reset for next loop iteration
                probs[i_k] = - torch.det(GG_num) / torch.det(GG_denom)
            else: # use block determinant formula
                t0 = time()
                Ksites_add = list(range(self.xmin, i_k+1))
                len_Ksites_add = i_k+1 - self.xmin
                occ_vec_add = torch.tensor([0] * (i_k - self.xmin) + [1])
                NN = torch.diag(occ_vec_add[:])

                if mm==0:
                    DDsubmat[mm, mm] =  idx_linearly_stored_G(self.G_lin_mem, Ksites_add, Ksites_add, chunk="D", 
                                    lr=len_Ksites_add, lc=len_Ksites_add) # self.G[np.ix_(Ksites_add, Ksites_add)]
                    assert torch.isclose(DDsubmat[mm, mm], self.G[np.ix_(Ksites_add, Ksites_add)]).all()    
                    if self.len_Ksites > 0: # How to deal with empty arrays ? 
                        BBsubmat[0:self.len_Ksites, 0:mm+1] = idx_linearly_stored_G(self.G_lin_mem, self.Ksites, Ksites_add, chunk="B", 
                                    lr=self.len_Ksites, lc=len_Ksites_add)                                   

                else:
                    A1 = DDsubmat[0:mm, 0:mm] # The A1-block stays constant. 
                    B1 = idx_linearly_stored_G(self.G_lin_mem, Ksites_add[:-1], [Ksites_add[-1]], chunk="B",
                                    lr=len_Ksites_add-1, lc=1)  # B1 = self.G[np.ix_(Ksites_add[:-1], [Ksites_add[-1]])]
                    assert torch.isclose(B1, self.G[np.ix_(Ksites_add[:-1], [Ksites_add[-1]])]).all()
                    C1 = idx_linearly_stored_G(self.G_lin_mem, [Ksites_add[-1]], Ksites_add[:-1], chunk="C",
                                    lr=1, lc=len_Ksites_add-1)    # C1 = self.G[np.ix_([Ksites_add[-1]], Ksites_add[:-1])]
                    assert torch.isclose(C1, self.G[np.ix_([Ksites_add[-1]], Ksites_add[:-1])]).all()
                    D1 = idx_linearly_stored_G(self.G_lin_mem, [Ksites_add[-1]], [Ksites_add[-1]], chunk="D",
                                    lr=1, lc=1)   # D1 = self.G[Ksites_add[-1], Ksites_add[-1]][None, None]  # convert tensor element to 2d array 
                    assert torch.isclose(D1, self.G[Ksites_add[-1], Ksites_add[-1]][None, None]).all()

                    ### DDsubmat = torch.vstack((torch.hstack((A1, B1)), torch.hstack((C1, D1))))  
                    ### DDsubmat[0:mm, 0:mm] = A1 # The A1-block stays constant.
                    DDsubmat[0:mm, mm] = B1[:, 0]
                    DDsubmat[mm, 0:mm] = C1[0, :]
                    DDsubmat[mm, mm] = D1
                    assert torch.isclose(DDsubmat[0:mm+1, 0:mm+1], torch.vstack((torch.hstack((A1, B1)), torch.hstack((C1, D1))))).all()

                    A2 = BBsubmat[:, 0:mm] # The A2-block stays constant

                    B2 = idx_linearly_stored_G_blockB1(self.G_lin_mem, self.Ksites, [Ksites_add[-1]], chunk="B1",
                                lr=self.len_Ksites, lc=1)  # B2 = self.G[np.ix_(self.Ksites, [Ksites_add[-1]])]

                    ###BBsubmat = torch.hstack((A2, B2))                    
                    ###BBsubmat[0:self.len_Ksites, 0:mm] = A2[:,0:mm] # The A2-block stays constant 
                    BBsubmat[0:self.len_Ksites, mm] = B2[:, 0]  
                    assert torch.isclose(BBsubmat[0:self.len_Ksites, 0:mm+1], torch.hstack((A2, B2))).all()

                t1 = time()
                self.t_fetch_memory += (t1 - t0)                    

                DD = DDsubmat[0:mm+1, 0:mm+1] - NN  # self.G[np.ix_(Ksites_add, Ksites_add)] - NN  
                BB = BBsubmat[0:self.len_Ksites, 0:mm+1]  # self.G[np.ix_(self.Ksites, Ksites_add)]      
                if self.len_Ksites == 0:                   
                   CC = BB
                else:
                   CC = BB.transpose(-1, -2)
                self.BB_reuse.append(BB)
                if self.state_index==-1: # no sampling step so far 
                    # here self.Xinv = [] always. IMPROVE: This line is useless.
                    # self.Xinv = torch.linalg.inv(self.G[np.ix_(self.Ksites, self.Ksites)] - torch.diag(torch.tensor(self.occ_vec[0:self.xmin])))   
                    self.Xinv = [] 
                else:
                    pass # self.Xinv should have been updated by calling update_state(pos_i)
                if self.len_Ksites == 0:                 
                   CCXinvBB[0:mm+1, 0:mm+1] = 0.0 
                   Schur_complement = DD - 0.0
                   detSC = torch.det(Schur_complement) # scalar                 
                else:
                   # Note: self.Xinv is a symmetric matrix and DD is also symmetric.
                   #       Thus, the Schur complement is also a symmetric matrix.            

                   # ------------------------------------------------------------------------------
                   # Iterative calculation of the Schur complement 
                   ##### original expression 
                   # self.Schur_complement = DD - torch.matmul(torch.matmul(CC, self.Xinv), self.BB)  
                   #####
                   if mm==0:     
                      t0 = time()
                      BtXinv = torch.matmul(CC, self.Xinv)              
                      CCXinvBB[0:mm+1, 0:mm+1] = torch.matmul(BtXinv, BB)
                      Schur_complement = DD - CCXinvBB[0:mm+1, 0:mm+1]
                      detSC = torch.det(Schur_complement) # scalar 

                      # The following is needed for iterative calculation of the determinant of  
                      # the Schur complement 
                      # `SCm_inv` is the inverse of the "modified Schur complement" where modified                       
                      # means that the lower right element of the Schur complement matrix is changed. 
                      # `SCm_inv` is updated iteratively using the formula for the inverse of a block matrix. 
                      SCm_inv = 1.0 / (Schur_complement + 1) #torch.linalg.inv(Schur_complememt + 1)
                      t1 = time()                      
                      self.t_update_Schur += (t1 - t0)
                   else:                      
                      t0 = time() 
                      AA1 = self.CCXinvBB_reuse[mm-1]         
                      BB1 = torch.matmul(BtXinv, BB[:,-1][:, None])
                      ###CC1 = BB1.transpose(-1,-2)
                      # update BtXinv = B.T * Xinv:
                      ###BtXinv = torch.vstack((BtXinv, torch.matmul(self.Xinv, BB[:,-1][:,None]).transpose(-1,-2)))                      
                      BtXinv_new[0:mm, :] = BtXinv
                      BtXinv_new[mm, :] = torch.matmul(self.Xinv, BB[:,-1])
                      assert torch.isclose(BtXinv_new[0:mm+1, :], torch.vstack((BtXinv, torch.matmul(self.Xinv, BB[:,-1][:,None]).transpose(-1,-2)))).all()
                      BtXinv = BtXinv_new[0:mm+1, :]
                      ####BtXinv = torch.vstack((BtXinv, torch.matmul(CC[-1,:][None,:], self.Xinv)))
                      DD1 = torch.matmul(BtXinv[-1,:][None,:], BB[:,-1][:,None])
                      ###CCXinvBB = torch.vstack((torch.hstack((AA1, BB1)), torch.hstack((CC1, DD1))))
                      CCXinvBB[0:mm, 0:mm] = AA1
                      CCXinvBB[0:mm, mm] = BB1[:,0]
                      CCXinvBB[mm, 0:mm] = BB1[:,0]
                      CCXinvBB[mm, mm] = DD1
                      ###assert torch.isclose(CCXinvBB, torch.vstack((torch.hstack((AA1, BB1)), torch.hstack((CC1, DD1))))).all()
                      Schur_complement = DD - CCXinvBB[0:mm+1, 0:mm+1]           
                      t1 = time()                      
                      self.t_update_Schur += (t1 - t0)
                      assert torch.isclose(Schur_complement, DD - torch.matmul(torch.matmul(CC, self.Xinv), BB)).all()                      
                      ### original formula: determinant of Schur complement ("SC")
                      ###detSC = torch.det(Schur_complement)
                      ###

                      t0 = time()
                      # calculate last element of SC_inv (inverse of the Schur complement) from SCm_inv (inverse of 
                      # modified Schur complement, which is being updated)
                      SC_inv_el =  SCm_inv[-1,-1]/(1.0 - SCm_inv[-1,-1])
                      detSC = detSC * (1.0 + SC_inv_el) \
                           * (Schur_complement[-1,-1] - torch.matmul(Schur_complement[-1,0:-1][None,:], \
                                                             torch.matmul(SCm_inv, Schur_complement[0:-1,-1][:,None])))
                      t1 = time()
                      self.t_matmul += (t1 - t0)  

                      assert torch.isclose(detSC, torch.det(Schur_complement))

                      # for next step of iterative calculation of the determinant of the Schur complement 
                      assert DD1.shape[0] == DD1.shape[1] == 1
                      t0 = time()
                      SCm_inv = block_update_inverse(SCm_inv, Schur_complement[0:-1,-1][:,None], \
                                Schur_complement[-1,0:-1][None,:], Schur_complement[-1,-1][None,None] + 1.0)    
                      t1 = time()                 
                      self.t_det += (t1 - t0)  
                                           
                   # ------------------------------------------------------------------------------                                   
                   #print("self.Xinv.shape=", self.Xinv.shape, "self.BB.shape=", self.BB.shape)                                     
                self.Schur_complement_reuse.append(Schur_complement)  
                self.CCXinvBB_reuse.append(CCXinvBB[0:mm+1, 0:mm+1])       
                            
                probs[i_k] = (-1) * detSC       
                cumul_probs += probs[i_k]

            # Finally, check whether the conditional probabilities are already saturated.
            # This gives significant speedup, most notably at low filling. 
            if cumul_probs > self.eps_norm_probs: #0.99999999:
                # print("skipping: sum(probs) already saturated (i.e. =1). xmax-1 - i_k = ", self.xmax-1 - i_k)
                probs[i_k+1:] = 0.0
                break
            #   
            # UNCOMMENTED BECAUSE OF ERROR MESSAGE
            # Error message: BB_ = self.BB_reuse[mm]      
            # # select previously computed matrices for the sampled position (i.e. pos_i, which is the mm-th position of the current support)
                               


        if not self.naive_update:
            assert len(self.BB_reuse) == len(self.Schur_complement_reuse) == (mm+1) # IMPROVE: remove this as well as the variable mm
        # IMPROVE: For large matrices the ratio of determinants leads to numerical
        # instabilities which results in not normalized probability distributions   
        # => use LOW-RANK UPDATE     
        #print("probs[:].sum()=", probs[:].sum()) 
        assert torch.isclose(probs.sum(), torch.tensor([1.0])) # assert normalization 
        # clamp negative values which are in absolute magnitude below machine precision
        probs = torch.where(abs(probs) > 1e-15, probs, torch.tensor([0.0]))

        return probs 


    def _sample_k(self, k):
        assert k < self.N 
        assert self.state_index == k-1

        probs = self.get_cond_prob(k)
        self.cond_probs[k*self.D:(k+1)*self.D] = probs
        #print("sum=", sum(self.cond_probs[k*self.D:(k+1)*self.D]))
        pos = Categorical(probs).sample().numpy()

        # conditional prob in this sampling step 
        cond_prob_k = probs[pos]

        # update state of the sampler given the result of the sampling step 
        self.update_state(pos.item())

        return pos, cond_prob_k

    #@profile
    def sample(self):
        self.reset_sampler()

        prob_sample = 1.0
        for k in np.arange(0, self.N):
            _, cond_prob_k = self._sample_k(k)
            prob_sample *= cond_prob_k
        
        np.savetxt("cond_probs_allk.dat", self.cond_probs.transpose())

        return self.occ_vec, prob_sample

    #@profile
    def update_state(self, pos_i):

        assert type(pos_i) == int 
        assert( 0 <= pos_i < self.D )
        k = self.state_index + 1

        self.Ksites.extend(list(range(self.xmin, pos_i+1)))
        self.len_Ksites += (pos_i+1 - self.xmin)
        self.occ_vec[self.xmin:pos_i] = 0
        self.occ_vec[pos_i] = 1
        self.occ_positions[k] = pos_i
        if not self.naive_update:
            # Update Xinv based on previous Xinv using 
            # formula for inverse of a block matrix 
            if self.state_index == -1: # first update step 
                Ksites_add = list(range(0, pos_i+1))
                occ_vec_add = [0] * pos_i + [1]
                if pos_i > 0: # np.diag() does not work for one-element arrays 
                    NN = torch.diag(torch.tensor(occ_vec_add[:]))
                else:
                    NN = torch.tensor(occ_vec_add)
                Ablock = self.G[np.ix_(Ksites_add, Ksites_add)] 
                Xinv_new = torch.linalg.inv(Ablock - NN)  
                self.Xinv = Xinv_new
            else:                
                Ksites_add = list(range(self.xmin, pos_i+1))  # put the sampled pos_i instead of loop variable i_k
                occ_vec_add = [0] * (pos_i - self.xmin) + [1] # IMPROVE: pos_i is a tensor here, which is not necessary 
                NN = torch.diag(torch.tensor(occ_vec_add[:]))

                mm = pos_i - self.xmin       # the mm-th position on the current support of the conditional probs
                BB_ = self.BB_reuse[mm]      # select previously computed matrices for the sampled position (i.e. pos_i, which is the mm-th position of the current support)
                Schur_complement_ = self.Schur_complement_reuse[mm]
                # Note: The Schur complement is a symmetric matrix, therefore Sinv is also symmetric. 
                XinvB = torch.matmul(self.Xinv, BB_) 
                Sinv = torch.linalg.inv(Schur_complement_)
                # IMPROVE: isn't this an outer product ? 
                Ablock = (self.Xinv + 
                    torch.matmul(torch.matmul(XinvB, Sinv), XinvB.transpose(-1,-2)))
                Bblock = - torch.matmul(XinvB, Sinv)
                # The following line is not needed due to symmetry since Cblock = Bblock.transpose(-1,-2).
                # Cblock = - torch.matmul(Sinv, XinvB.transpose(-1,-2))
                Cblock = Bblock.transpose(-1,-2)
                Dblock = Sinv 
                Xinv_new = torch.vstack(( torch.hstack((Ablock, Bblock)), torch.hstack((Cblock, Dblock)) ))  # np.block([[Ablock, Bblock], [Cblock, Dblock]])
                self.Xinv = Xinv_new

                # REMOVE
                cond = np.linalg.cond(Xinv_new)
                if cond > self.cond_max:
                    self.cond_max = cond
                    print("cond_max=", self.cond_max)


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
        assert row_idx.shape[-1] == self.P_ortho.shape[-1]
        # select 2 (3,4,...) rows from a matrix with 2 (3,4,...) columns 
        # submat = np.take(self.P, row_idx, axis=-2) # broadcast over leading dimensions of row_idx
        submat = self.P_ortho[..., row_idx, :]
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

    from time import time 

    for L in (200,): #(1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000):
        (Nsites, eigvecs) = prepare_test_system_zeroT(Nsites=L, potential='none', PBC=True, HF=True)
        Nparticles = 100 #L//2
        num_samples = 5 # 1000

        #SDsampler  = SlaterDetSampler_ordered(Nsites=Nsites, Nparticles=Nparticles, single_particle_eigfunc=eigvecs, naive_update=True)
        SDsampler1 = SlaterDetSampler_ordered(Nsites=Nsites, Nparticles=Nparticles, single_particle_eigfunc=eigvecs, naive_update=True)
        SDsampler2 = SlaterDetSampler_ordered(Nsites=Nsites, Nparticles=Nparticles, single_particle_eigfunc=eigvecs, naive_update=False)

        print("Nsites=", Nsites, "Nparticles=", Nparticles, "num_samples=", num_samples)

        t0 = time()
        for _ in range(num_samples):
            occ_vec, _ = SDsampler1.sample()
        t1 = time()
        print("naive, elapsed=", (t1-t0) )

        t0 = time()
        for _ in range(num_samples):
            occ_vec, prob_sample = SDsampler2.sample()
            print("prob_sample=", prob_sample)
            print("log(prob_sample)=", np.log(prob_sample))
        t1 = time()
        print("block update, elapsed=", (t1-t0) )

        print("t_fetch_memory=", SDsampler2.t_fetch_memory)
        print("t_matmul(Schur complement)=", SDsampler2.t_matmul)
        print("t_det=", SDsampler2.t_det)
        print("t_update_Schur=", SDsampler2.t_update_Schur)

    # # Check that sampling the Slater determinant gives the correct average density. 
    # occ_vec = torch.zeros(Nsites)
    # for s in range(num_samples):
    #     occ_vec_, prob_sample = SDsampler2.sample()
    #     print("=================================================================")
    #     print("amp_sample= %16.8f"%(np.sqrt(prob_sample)))
    #     print("naive sampler: amplitude= %16.8f"%(SDsampler1.psi_amplitude(occ_vec_)))
    #     print("block update sampler: amplitude=", SDsampler2.psi_amplitude(occ_vec_))
    #     print("=================================================================")
    #     occ_vec += occ_vec_
       

    # #print("occ_vec=", occ_vec)    
    # density = occ_vec / float(num_samples)
    # #print("density=", density)

    # OBDM = Slater2spOBDM(eigvecs[:, 0:Nparticles])

    # f = plt.figure(figsize=(8,6))
    # ax = f.add_subplot(1,1,1)
    # ax.set_xlabel(r"site $i$")
    # ax.set_ylabel(r"av. density")
    # ax.plot(range(len(density)), density, label=r"av.density $\langle n \rangle$ (sampled)")
    # ax.plot(range(len(np.diag(OBDM))), np.diag(OBDM), label=r"$\langle n \rangle$ (from OBDM)")
    # plt.legend()
    # #plt.show()
