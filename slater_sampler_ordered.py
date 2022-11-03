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

# for lowrank_kinetic()
from time import time 
from k_copy import calc_k_copy
from monitoring_old import logger as mylogger
import lowrank_update as LR

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


    def _detratio_from_scratch_v0(self, G, occ_vec, base_pos, i):
        """
        Calculate ratio of determinants of numerator and denominator matrices from scratch.    
        """
        base = list(range(0, base_pos+1))
        occ_vec_base = list(occ_vec[0:base_pos + 1])
        Gdenom = G[np.ix_(base, base)] - np.diag(occ_vec_base)
        extend = list(range(0, i+1))
        occ_vec_add = [1] if (i - base_pos == 1) else [0] * (i - base_pos - 1) + [1]
        occ_vec_extend = occ_vec_base + occ_vec_add
        Gnum = G[np.ix_(extend, extend)] - np.diag(occ_vec_extend)

        sign_num, logdet_num = np.linalg.slogdet(Gnum)
        sign_denom, logdet_denom = np.linalg.slogdet(Gdenom)

        ratio = sign_num * sign_denom * np.exp(logdet_num - logdet_denom)
        # return np.linalg.det(Gnum) / np.linalg.det(Gdenom)
        return ratio 


    def _detratio_from_scratch_tmp(self, G, occ_vec, base_pos, i):
        """
        Calculate ratio of determinants of numerator and denominator matrices from scratch.    
        """
        from scipy import linalg  # scipy.linalg supports float128, np.linalg does not 
        G = np.array(G, dtype=np.float128)  # use quadruple precision 

        base = list(range(0, base_pos+1))
        occ_vec_base = list(occ_vec[0:base_pos + 1])
        Gdenom = G[np.ix_(base, base)] - np.diag(occ_vec_base)
        extend = list(range(0, i+1))
        occ_vec_add = [1] if (i - base_pos == 1) else [0] * (i - base_pos - 1) + [1]
        occ_vec_extend = occ_vec_base + occ_vec_add
        Gnum = G[np.ix_(extend, extend)] - np.diag(occ_vec_extend)

        ratio = linalg.det(Gnum) / linalg.det(Gdenom)
        return ratio 


    #@profile 
    def _detratio_from_scratch(self, G, occ_vec, base_pos, i):
        """
        Calculate ratio of determinants of numerator and denominator matrices from scratch.  
        Use singular value decomposition to separate scales in matrix-matrix multiplication.   
        Cancel the denominator determinant: det(Gdenom)
        """
        from scipy import linalg 
        occ_vec_base = list(occ_vec[0:base_pos + 1])
        occ_vec_add = [1] if (i - base_pos == 1) else [0] * (i - base_pos - 1) + [1]
        K1 = list(range(0, base_pos+1))
        K2 = list(range(base_pos+1, i+1))
        X = G[np.ix_(K1, K1)] - np.diag(occ_vec_base)
        B = G[np.ix_(K1, K2)]
        C = B.transpose()
        D = G[np.ix_(K2, K2)] - np.diag(occ_vec_add)

        # use singular value decomposition to separate scales     
        # cancel the denominator matrix        
        uu, ss, vv = np.linalg.svd(X)
        # THIS DOES NOT REALLY WORK because large scales in 1/ss still swamp the smaller scales. 
        # Pseudo-inverse 
        ratio = np.linalg.det( D - ( (C @  vv.T) @ np.diag(np.where(ss>LR.thresh, 1.0/ss, 0.0)) @ (uu.T @ B) ) )

        #if __debug__:
        #    occ_vec_extend = occ_vec_base + occ_vec_add
        #    extend = list(range(0, i+1))
        #    G = np.array(G, dtype=np.float64) # np.float128
        #    Gnum = G[np.ix_(extend, extend)] - np.diag(occ_vec_extend)
        #    Gdenom = G[np.ix_(K1, K1)] - np.diag(occ_vec_base)
        # 
        #    ratio2 = linalg.det(Gnum) / linalg.det(Gdenom)
        #
        #    # print("ratio2=", ratio2, "ratio=", ratio, "condition number num, denom, X=", np.linalg.cond(Gnum), np.linalg.cond(Gdenom), np.linalg.cond(X))

        return ratio 


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
        
    #@profile
    def lowrank_kinetic(self, ref_I, xs_I, rs_pos, print_stats=True):
        """
        Probability density estimation on states connected to I_ref by the kinetic operator `kinetic_operator`,
        given the Slater-Jastrow ansatz. 
        
        This routine is similar to sampling, but not quite. It duplicates much of the sampling routine of the 
        Slater sampler. 

        Parameters:
        -----------
        I_ref : int
            bitcoded integer of the reference state 
        kinetic_operator : func 
            (I_ref, lattice_object) -> (hop_from_to, I_prime, matrix_elem)

        Returns:
        --------
        All conditional probabilities for the onehop states. 
        
        (In another routine, they are to be multiplied by the conditional 
        probabilities coming from MADE. -> normalize -> obtain cond. probs. at actually sampled positions
        -> ratios <beta|psi> / <alpha|psi> -> local kinetic energy for state |alpha> (no backpropagation required). )
        """

        GG = self.G.detach().numpy()

        assert_margin = 1e-6

        # normalization needs to be satisfied up to 
        #     \sum_i p(i)  > `eps_norm_probs``
        eps_norm_probs = 1.0 - 1e-10


        def _copy_cond_probs(cond_prob_ref, cond_prob_onehop, one_hop_info):
            """
            Copy all conditional probabilities for k <= k_copy_, which are identical in the reference 
            state and in the one-hop states.
            """
            for state_nr, (k_copy, _) in enumerate(one_hop_info):
                cond_prob_onehop[state_nr, 0:k_copy+1, :] = cond_prob_ref[0:k_copy+1, :]


        num_onehop_states = len(xs_I)
        xs = int2bin(xs_I, ns=self.D)
        ref_conf = int2bin(ref_I, ns=self.D)    
        mylogger.info_refstate.num_onehop_states = num_onehop_states
        #mylogger.info_refstate.accumulator["num_onehop_states"] = num_onehop_states

        # special case of 1d n.n. hopping matrix
        # assert np.all([abs(r-s) == 1 or abs(r-s) == Ns-1 for r,s in rs_pos])

        k_copy = calc_k_copy(rs_pos, ref_conf)
        onehop_info = list(zip(k_copy, rs_pos))

        # For s < r, the support for conditional probabilities in the onehop state
        # is larger than in the reference state: xmin(conn) < xmin(ref). 
        # There the probabilities for additional sites need to be calculated which have no counterpart 
        # in the calculations for the reference state. 
        # SK_s_lt_r = ((s0, k0), (s1, k1), ...) where the pair (si, ki) means: ki-th particle sits at position si.
        # Only pairs with s < r are included. 
        s_pos = list(s for (r,s) in rs_pos if s < r)
        SK_s_lt_r = list(zip(s_pos, [kk for idx, kk in enumerate(k_copy) if rs_pos[idx][1] < rs_pos[idx][0]]))
        det_Gnum_reuse = dict()

        cond_prob_ref = np.zeros((self.N, self.D))
        cond_prob_onehop = np.zeros((num_onehop_states, self.N, self.D))
        cumsum_condprob_onehop = np.zeros((num_onehop_states, self.N))

        # The following variables are needed for low-rank update for onehop states differing 
        # from the reference state by long-range hopping between positions r and s.
        Gdenom_inv_reuse = dict()
        det_Gdenom_reuse = dict()
        Gnum_inv_reuse = dict()


        Ksites = []
        occ_vec = list(ref_conf)
        assert type(occ_vec) == type(list()) # use a list, otherwise `occ_vec[0:xmin] + [1]` will result in `[]`. 
        pos_vec = bin2pos(ref_conf)

        xs_pos = bin2pos(xs)

        # Needed for long-range hopping in 2D. 
        # At position `s` (`r`) sits the `k_s[state_nr]`-th (`k_r[state_nr]`-th) particle. 
        # Here, k_s and k_r are counted in the the onehop state. However, the loop index k 
        # is referring to the particle numbers in the reference state. 
        k_s = [np.searchsorted(xs_pos[state_nr, :], rs_pos[state_nr][1]) for state_nr in range(num_onehop_states)]
        k_r = [np.searchsorted(xs_pos[state_nr, :], rs_pos[state_nr][0]) for state_nr in range(num_onehop_states)]

        for k in range(self.N):
            # Calculate the conditional probabilities for the k-th particle (for all onehop states 
            # connected to the reference state through the kinetic operator simultaneously, using a low-rank update).
            xmin = 0 if k==0 else pos_vec[k-1] + 1 # half-open interval (xmin included, xmax not included)
            xmax = self.D - self.N + k + 1
            Ksites = list(range(0, xmin))
            Ksites_add = Ksites.copy()

            Gnum_inv_reuse[k] = dict.fromkeys(range(xmin, xmax))

            if k >= 2:
                # don't waste memory
                Gnum_inv_reuse[k-2].clear()

            t1 = time()
            Gdenom = GG[np.ix_(Ksites, Ksites)] - np.diag(occ_vec[0:len(Ksites)])
            t2 = time()
            self.t_npix_ += (t2 - t1)

            # In production runs use flag -O to suppress asserts and 
            # __debug__ sections. 
            if __debug__:
                if Gdenom.shape[0] > 0:
                    cond = np.linalg.cond(Gdenom)
                    if cond > mylogger.info_refstate.Gdenom_cond_max:
                        mylogger.info_refstate.Gdenom_cond_max = cond 
                        if cond > 1e5:
                            print("Gdenom_cond_max=", mylogger.info_refstate.Gdenom_cond_max)            

            t1 = time()
            det_Gdenom = np.linalg.det(Gdenom)

            # Internal state used during low-rank update of conditional probabilities 
            # of the connnecting states. 
            Gdenom_inv = np.linalg.inv(Gdenom)
            t2 = time()
            self.t_lowrank_linalg += (t2 -t1)

            # Needed for low-rank update for onehop states differing from the reference 
            # state by long-range hopping between positions r and s. 
            # (It is important the quantities that are to be reused are only taken from the reference 
            # state since quantities taken from a onehop state would be overwritten by other onehop states.)
            Gdenom_inv_reuse[k] = Gdenom_inv # does not depend on i 
            det_Gdenom_reuse[k] = det_Gdenom # does not depend on i

            for ii, i in enumerate(range(xmin, xmax)):
                t0=time()
                # reference state        
                Ksites_add += [i]
                occ_vec_add = occ_vec[0:xmin] + [0]*ii + [1]
                t1 = time()
                Gnum = GG[np.ix_(Ksites_add, Ksites_add)] - np.diag(occ_vec_add)
                t2 = time()
                self.t_npix_ += (t2-t1)
                
                t6 = time()
                det_Gnum = np.linalg.det(Gnum)
                t7 = time()
                self.t_lowrank_linalg += (t7-t6)

                # In case a cond. prob. of the reference state is zero:
                try:
                    t6 = time()
                    Gnum_inv = np.linalg.inv(Gnum)
                    t7 = time()
                    self.t_lowrank_linalg += (t7-t6)
                    Gnum_inv_reuse[k][i] = Gnum_inv
                except np.linalg.LinAlgError as e:
                    print("Cond. prob. of reference state is zero: det_Gnum=%16.12f\n" % (det_Gnum), e)
                    # Since the matrix inversion failed, Gnum_inv_reuse[k][i] == None. 
                    assert Gnum_inv_reuse[k][i] is None
                    # This will be checked for later before reusing Gnum_in_reuse[k][i]. 
                
                cond_prob_ref[k, i] = (-1) * det_Gnum / det_Gdenom
                t1 = time() 
                mylogger.info_refstate.elapsed_ref += (t1 - t0)
                #mylogger.info_refstate.accumulator["elapsed_ref"] = (t1 - t0)

                if (i,k) in SK_s_lt_r:
                    det_Gnum_reuse.update({k : det_Gnum})

                # Now calculate the conditional probabilities for all states related 
                # to the reference state by one hop, using a low-rank update of `Gnum`
                # and `Gdenom`.
                t0_conn = time()
                for state_nr, (k_copy_, (r,s)) in enumerate(onehop_info):
                    if k_copy_ >= k:
                        # Copy conditional probabilities rather than calculating them.
                        # Exit the loop; it is assumed that onehop states are ordered according to increasing values 
                        # of k_copy_, i.e. this condition is also fulfilled for all subsequent onehop states.
                        break 
                    else: # k_copy < k  
                        xmin_onehop = xs_pos[state_nr, k-1] + 1; xmax_onehop = xmax                         
                        if ii == 0:
                            mylogger.info_refstate.size_support += xmax_onehop - xmin_onehop
                        # # SOME SPECIAL CASES 
                        if xmin_onehop == xmax_onehop-1 and i == xmin_onehop: #and r < i and s < i: # and (r < s and s < i) or (s < r and r < i): #
                            #print("certainly 1")
                            # print("r=", r, "s=", s)
                            #print("xmin_onehop=", xmin_onehop, "xmax_onehop=", xmax_onehop)
                            #print("k=", k, "i=", i, "state_nr=", state_nr)
                            #print(ref_conf)
                            #print(xs[state_nr])
                            #print(" xs_pos[state_nr, k-1]=",  xs_pos[state_nr, k-1])
                            # exit(1)
                        # In this case it is clear that every subsequent empty site needs to be occupied to accomodate 
                        # all particles both in the reference state and in the one-hop state. Don't calculate probabilities. 
                            cond_prob_onehop[state_nr, k, i] = 1.0
                            cumsum_condprob_onehop[state_nr, k] = 1.0
                            continue

                        if abs(r-s) >= 1: # long-range hopping in 1d (This does not include special cases for long-range hopping due to 2D geometry.)

                            # # First check whether the conditional probabilities are already saturated.
                            # NOTE: The cond. prob. at the actually sampled positions needs to be computed before 
                            #       saturation of the normalization can be exploited.
                            # IMPROVE: make sure that all subsequent `i` (for given state_nr and k) are automatically skipped without 
                            # testing this conditions again `
                            if cumsum_condprob_onehop[state_nr, k] > eps_norm_probs and i > xs_pos[state_nr, k]:  
                                cond_prob_onehop[state_nr, k, i:] = 0.0
                                mylogger.info_refstate.counter_skip += 1 #(xmax - i)
                                #mylogger.info_refstate.accumulator["counter_skip"] = (xmax - i)  
                                continue

                            if r < s:
                                if k > 1 and k <= k_s[state_nr]: # k=0 can always be copied from the reference state
                                    if Gnum_inv_reuse[k][i] is not None:
                                        try:
                                            corr_factor = LR.remove_r(Gnum_inv_reuse[k][i], Gdenom_inv_reuse[k], r=r)
                                            # NOTE: The cond. probs. for (k-1)-th particle are computed retroactively while the 
                                            # cond. probs. for k-th particle of the reference state are being computed. 
                                            cond_prob_onehop[state_nr, k-1, i] = corr_factor * cond_prob_ref[k, i] # update: (k-1) -> k                                                
                                        except LR.ErrorFinitePrecision as e:
                                            print("Excepting finite precision error 1, state_nr=", state_nr, "k=", k, "i=", i, "msg=", e)
                                            cond_prob_onehop[state_nr, k-1, i] = (-1) * self._detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-2], i=i)                                             
                                    else:
                                        print("Gnum_inv_reuse is None (i.e. accidental zero cond. prob. of reference state)")
                                        cond_prob_onehop[state_nr, k-1, i] = (-1) * self._detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-2], i=i)                                                                             
                                    assert -assert_margin <= cond_prob_onehop[state_nr, k-1, i] <= 1.0 + assert_margin, "cond prob = %16.10f" %(cond_prob_onehop[state_nr, k-1, i])


                                # additionally ...
                                ###if k == self.N-1 or k == k_s[state_nr]: 
                                if k == k_s[state_nr]:     
                                    # Special case: cond. probs. for last particle (for 1D system with pbc) or 
                                    # last particle whose support involves position `s` (for 2D system).      
                                    if i > xs_pos[state_nr, k-1]: # support is smaller than in the reference state      
                                        if Gnum_inv_reuse[k][xs_pos[state_nr, k-1]] is not None:                               
                                            try:
                                                Gnum_inv_, corr1 = LR.adapt_Ainv(Gnum_inv_reuse[k][xs_pos[state_nr, k-1]], Gglobal=GG, r=r, s=s, i_start=xs_pos[state_nr, k-1]+1, i_end=i)                                       
                                                corr2 = LR.corr_factor_remove_r(Gnum_inv_, r=r)                                               
                                                corr_factor_Gnum = corr1 * corr2                                     
                                                Gdenom_inv_ = Gnum_inv_reuse[k][xs_pos[state_nr, k-1]]
                                                corr_factor_Gdenom = LR.corr_factor_remove_r(Gdenom_inv_, r=r) * ( det_Gnum / det_Gdenom )
                                                corr_factor = corr_factor_Gnum / corr_factor_Gdenom 
                                                cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]                                                      
                                            except np.linalg.LinAlgError as e: # from inverting singular matrix in LR.adapt_Ainv()
                                                print("Excepting LinAlgError 1, state_nr=", state_nr, "k=", k, "i=", i, "msg=", e)
                                                cond_prob_onehop[state_nr, k, i] = (-1) * self._detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                            except LR.ErrorFinitePrecision as e:
                                                print("Excepting LR.ErrorFinitePrecision 1, state_nr=", state_nr, "k=", k, "i=", i, "msg=", e)
                                                cond_prob_onehop[state_nr, k, i] = (-1) * self._detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                        else:
                                            print("Gnum_inv_reuse is None (i.e. accidental zero cond. prob. of reference state)")
                                            cond_prob_onehop[state_nr, k, i] = (-1) * self._detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                        assert -assert_margin <= cond_prob_onehop[state_nr, k, i] <= 1.0 + assert_margin, "cond_prob=%16.10f" % (cond_prob_onehop[state_nr, k, i])

                                # Yet another special case (only relevant in 2D since there i > s is possible)
                                # k is the component in the reference state 
                                if k == k_s[state_nr] + 1 and i > s:
                                    # For the moment, calculate the ratio of determinants from scratch. 
                                    # Compared to the next case, here, the denominator matrix needs to be adjusted relative 
                                    # to the reference state (i.e. it is not simply a correction factor). 
                                    # IMPROVE: Design a lowrank update for this case. 

                                    # extend current Gdenom so as to include also a particle at position s
                                    try:
                                        corr_factor_Gnum = LR.corr_factor_removeadd_rs(Gnum_inv, r=r, s=s)
                                        Gdenom_inv_, corr1 = LR.adapt_Ainv(Gdenom_inv, Gglobal=GG, r=r, s=s, i_start=pos_vec[k-1]+1, i_end=s)
                                        corr2 = LR.corr_factor_remove_r(Gdenom_inv_, r=r)
                                        corr_factor_Gdenom = corr1 * corr2 
                                        if corr_factor_Gdenom < LR.thresh:
                                            raise LR.ErrorFinitePrecision
                                        corr_factor = corr_factor_Gnum / corr_factor_Gdenom                                        
                                        cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k,i]                                                                           
                                    except np.linalg.LinAlgError as e:
                                        print("Excepting LinAlgError 2, state_nr=", state_nr, "k=", k, "i=", i, "msg=", e)
                                        cond_prob_onehop[state_nr, k, i] = (-1) * self._detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                    except LR.ErrorFinitePrecision as e: #np.linalg.LinAlgError as e: # from inverting singular matrix in LR.adapt_Ainv() 
                                        print("Excepting LR.ErrorFinitePrecision 2, state_nr=", state_nr, "k=", k, "i=", i, "msg=", e)
                                        cond_prob_onehop[state_nr, k, i] = (-1) * self._detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                    assert -assert_margin <= cond_prob_onehop[state_nr, k, i] <= 1.0 + assert_margin, "cond_prob[state_nr=%d, k=%d, i=%d]=%16.10f" % (state_nr, k, i, cond_prob_onehop[state_nr, k, i])

                                # Yet another special case (only relevant in 2D)
                                elif k > k_s[state_nr] + 1 and i > s:
                                    try:
                                        corr_factor = LR.removeadd_rs(Gnum_inv, Gdenom_inv, r, s)
                                        cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]                                                                                                                     
                                    except LR.ErrorFinitePrecision as e: 
                                        print("Excepting finite precision error 2, state_nr=", state_nr, "k=", k, "i=", i, "msg=", e)
                                        cond_prob_onehop[state_nr, k, i] = (-1) * self._detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                    assert -assert_margin <= cond_prob_onehop[state_nr, k, i] <= 1.0 + assert_margin, "cond. prob = %16.10f" % (cond_prob_onehop[state_nr, k, i])
                            
                            elif r > s:
                                if i > s:
                                    if k <= k_r[state_nr]:
                                        if i==pos_vec[k-1]+1:
                                        # need to calculate some additional cond. probs. along the way 
                                            for j_add in range(xmin_onehop, pos_vec[k-1]+1):  
                                                try:
                                                    # reuse all information from (k-1) cond. probs. of reference state
                                                    if k == k_copy_ + 1 and s == 0: # 1D long-range hopping      
                                                        # IMPROVE: Gdenom_inv_ is not needed subsequently.                                        
                                                        # Gdenom_inv_, corr_factor_Gdenom2 = LR.adapt_Gdenom_inv(Gdenom_inv_reuse[k-1], Gglobal=GG, r=r, s=xs_pos[state_nr, k-1])
                                                        corr_factor_Gdenom = LR.corr_factor_Gdenom_from_Ainv(Gdenom_inv_reuse[k-1], Gglobal=GG, r=r, s=s, i_start=0, i_end=xs_pos[state_nr, k-1])
                                                        #print("corr_factor_Gdenom=", corr_factor_Gdenom)
                                                        #print("corr_factor_Gdenom2=", corr_factor_Gdenom2)
                                                    elif k == k_copy_ + 1 and s > 0: # 2D long-range hopping: There are particles or empty sites to the left of s.                                            
                                                        # 1. No particle to the left of position `s`. (Of course, this is so both in the reference state and in the onhop
                                                        # state since they differ only in the occupancies of the positions `s` and `r`.)
                                                        if k_s[state_nr] == 0:
                                                            # Gdenom_inv_, corr_factor_Gdenom = LR.adapt_Ainv(np.array([]).reshape(0,0), Gglobal=GG, r=r, s=s, i_start=0, i_end=s)
                                                            corr_factor_Gdenom = LR.corr_factor_Gdenom_from_Ainv(np.array([]).reshape(0,0), Gglobal=GG, r=r, s=s, i_start=0, i_end=s)
                                                        # 2. There is at least one particle to the left of position `s`. Numerator and denominator 
                                                        # matrices can be extended from that position. 
                                                        elif k_s[state_nr] > 0:
                                                            # We are calculating cond. probs. for the k-th particle. For the reference state low-rank updates 
                                                            # are based on the sampled position of the (k-1)-th particle. For the onehop state they are based 
                                                            # on the position of the (k-2)-th particle. 
                                                            i_start = pos_vec[k-2]+1 if k >= 2 else 0
                                                            #Gdenom_inv_, corr_factor_Gdenom = LR.adapt_Ainv(Gdenom_inv_reuse[k-1], Gglobal=GG, r=r, s=s, i_start=i_start, i_end=s)
                                                            corr_factor_Gdenom = LR.corr_factor_Gdenom_from_Ainv(Gdenom_inv_reuse[k-1], Gglobal=GG, r=r, s=s, i_start=i_start, i_end=s)
                                                    else:
                                                        corr_factor_Gdenom = LR.corr_factor_add_s(Gdenom_inv_reuse[k-1], s=s)

                                                    if Gnum_inv_reuse[k-1][j_add] is not None:                                            
                                                        corr_factor_Gnum = LR.corr_factor_add_s(Gnum_inv_reuse[k-1][j_add], s=s) # CAREFUL: This is not a marginal probability of an actually sampled state.
                                                        if not math.isclose(corr_factor_Gdenom, 0.0, abs_tol=1e-15): # don't divide by zero 
                                                            corr_factor = corr_factor_Gnum / corr_factor_Gdenom
                                                            cond_prob_onehop[state_nr, k, j_add] = corr_factor * cond_prob_ref[k-1, j_add]
                                                        else:
                                                            cond_prob_onehop[state_nr, k, j_add] = (-1) * self._detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=j_add)
                                                    else:
                                                        print("Gnum_inv_reuse is None (i.e. accidental zero cond. prob. of reference state)")
                                                        cond_prob_onehop[state_nr, k, j_add] = (-1) * self._detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=j_add)
                                                except LR.ErrorFinitePrecision as e:
                                                        cond_prob_onehop[state_nr, k, j_add] = (-1) * self._detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=j_add)                                                                
                                                assert -assert_margin <= cond_prob_onehop[state_nr, k, j_add] <= 1.0 + assert_margin, "cond. prob = %16.10f" % (cond_prob_onehop[state_nr, k, j_add])  
                                                # update cumul. probs. explicitly because this is inside the body of an extra loop                                              
                                                cumsum_condprob_onehop[state_nr, k] += cond_prob_onehop[state_nr, k, j_add]

                                        try:
                                            if k == k_copy_ + 1:
                                                if s==0:                                                                                       
                                                    #Gdenom_inv_, corr_factor_Gdenom = LR.adapt_Ainv(np.array([]).reshape(0,0), Gglobal=GG, r=r, s=s, i_start=0, i_end=s)
                                                    corr_factor_Gdenom = LR.corr_factor_Gdenom_from_Ainv(np.array([]).reshape(0,0), Gglobal=GG, r=r, s=s, i_start=0, i_end=s)
                                                    #print("state_nr=", state_nr, "k=", k, "i=", i, "corr_Gnum=", corr_factor_Gnum, "corr_Gdenom=", corr_factor_Gdenom)        
                                                elif s > 0:
                                                    i_start = pos_vec[k-2]+1 if k >= 2 else 0                                         
                                                    #Gdenom_inv_, corr_factor_Gdenom = LR.adapt_Ainv(Gdenom_inv_reuse[k-1], Gglobal=GG, r=r, s=s, i_start=i_start, i_end=s)
                                                    corr_factor_Gdenom = LR.corr_factor_Gdenom_from_Ainv(Gdenom_inv_reuse[k-1], Gglobal=GG, r=r, s=s, i_start=i_start, i_end=s)                                        
                                                    #print("state_nr=", state_nr, "k=", k, "i=", i, "corr_Gnum=", corr_factor_Gnum, "corr_Gdenom=", corr_factor_Gdenom)        
                                            else:                 
                                                corr_factor_Gdenom= LR.corr_factor_add_s(Gdenom_inv_reuse[k-1], s=s) 
                                            if not math.isclose(corr_factor_Gdenom, 0.0, abs_tol=1e-15): # do not divide by zero
                                                corr_factor_Gnum = LR.corr_factor_removeadd_rs(Gnum_inv, r=pos_vec[k-1], s=s)
                                                corr_factor = corr_factor_Gnum / corr_factor_Gdenom
                                                cond_prob_onehop[state_nr, k, i] = corr_factor * (det_Gdenom / det_Gdenom_reuse[k-1]) * cond_prob_ref[k, i]    
                                            else:
                                                cond_prob_onehop[state_nr, k, i] = (-1) * self._detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                        except LR.ErrorFinitePrecision as e:
                                            cond_prob_onehop[state_nr, k, i] = (-1) * self._detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                        assert -assert_margin <= cond_prob_onehop[state_nr, k, i] <= 1.0 + assert_margin, "cond prob = %16.10f" %(cond_prob_onehop[state_nr, k, i])

                                    elif k > k_r[state_nr]: # conditional probs. of reference state and onehop state have the same support 
                                        try:                                           
                                            corr_factor = LR.removeadd_rs(Gnum_inv, Gdenom_inv, r, s)
                                            cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]
                                        except LR.ErrorFinitePrecision as e:
                                            print("k>k_r: Excepting finite precision error 3, state_nr=", state_nr, "k=", k, "i=", i, "msg=", e)
                                            print("ref_conf    =", ref_conf)
                                            print("xs[state_nr]=", xs[state_nr])                                            
                                            cond_prob_onehop[state_nr, k, i] = (-1) * self._detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                        assert -assert_margin <= cond_prob_onehop[state_nr, k, i] <= 1.0 + assert_margin, "cond_prob[state_nr=%d, k=%d, i=%d]=%16.10f" % (state_nr, k, i, cond_prob_onehop[state_nr, k, i])
                                                            
                        #assert cond_prob_onehop[state_nr, k, i] >= -1e-8, "state_nr=%d, k=%d, i=%d, r=%d, s=%d" %(state_nr, k, i, r, s)
                        assert -assert_margin <= cond_prob_onehop[state_nr, k, i] <= 1.0 + assert_margin, "cond_prob=%16.10f" % (cond_prob_onehop[state_nr, k, i])
                        cumsum_condprob_onehop[state_nr, k] += cond_prob_onehop[state_nr, k, i]                    

                t1_conn = time()
                mylogger.info_refstate.elapsed_connecting_states += (t1_conn - t0_conn)
                #mylogger.info_refstate.accumulator["elapsed_connecting_states"] = (t1_conn - t0_conn)                    
                        
        _copy_cond_probs(cond_prob_ref, cond_prob_onehop, onehop_info)

        if print_stats:
            mylogger.info_refstate.print_summary()
        mylogger.info_refstate.reset()

        if False: #__debug__:
            fh = open("cond_prob_ref.dat", "w")
            fh.write("# ref_state ["+" ".join(str(item) for item in ref_conf)+"]\n\n")
            for k in range(cond_prob_ref.shape[0]):
                for i in range(cond_prob_ref.shape[1]):
                    fh.write("%d %d %20.19f\n" % (k, i, cond_prob_ref[k, i]))
            fh.close()

            for state_nr in range(cond_prob_onehop.shape[0]):
                fh = open("cond_prob_onehop%d.dat" % (state_nr), "w")
                fh.write("# ref_state ["+" ".join(str(item) for item in ref_conf)+"]\n")
                fh.write("# 1hop      ["+" ".join(str(item) for item in xs[state_nr])+"]\n")
                (k_copy, (r, s)) = onehop_info[state_nr]
                fh.write("# k_copy=%d, r=%r, s=%s \n" % (k_copy, r, s))
                fh.write("# ========================================\n")
                fh.write("# k | i | cond_prob(k,i) | IN_SUPPORT ?\n")

                for k in range(cond_prob_onehop.shape[1]):
                    for i in range(cond_prob_onehop.shape[2]):
                        IN_SUPPORT = i in np.arange(xs_pos[state_nr, k-1] + 1, self.D - self.N + k + 1)
                        fh.write("%d %d %20.19f %s\n" % (k, i, cond_prob_onehop[state_nr, k, i], IN_SUPPORT))
                fh.close()   

        # Check that all conditional probabilities are normalized. 
        # IMPROVE: Return also information about violation of probability normalization. 
        if True: #__debug__:
                for state_nr, (k_copy_, (r,s)) in enumerate(onehop_info):
                    for k in range(self.N):
                        assert math.isclose(np.sum(cond_prob_ref[k,:]), 1.0, abs_tol=1e-14), \
                            "np.sum(cond_prob_ref[k=%d,:])=%16.10f" % (k, np.sum(cond_prob_ref[k,:]))
                        if k > k_copy_:
                            # print("state_nr=", state_nr, "k=", k, "rs_pos[state_nr]=", rs_pos[state_nr])
                            # print("ref_conf=", ref_conf)
                            # print("onehop  =", xs[state_nr])
                            #print("cumsum_condprob_onehop[state_nr, k]=", cumsum_condprob_onehop[state_nr, k])
                            #print("state_nr=", state_nr, "k=", k, "cond_prob=", cond_prob_onehop[state_nr, k, :])
                            summe=np.sum(cond_prob_onehop[state_nr, k,:])
                            if not math.isclose(summe, 1.0, abs_tol=1e-8):
                                fh = open(self.dir+"NormalizationViolation.dat", "a")
                                fh.write("np.sum(cond_prob_onehop[state_nr=%d, k=%d])=%16.10f =? 1.0 =? %16.10f" \
                                 % (state_nr, k, summe, cumsum_condprob_onehop[state_nr, k])+"\n")
                                fh.close()                                                            
                                print("ERROR: NORMALIZATION VIOLATION, sum(cond_probs)="+str(summe)+"  - set cond. probs. equal to reference state")
                                print("CAREFUL ! THIS IS NOT JUSTIFIED")
                                cond_prob_onehop[state_nr, k, :] = cond_prob_ref[k, :]
                            assert math.isclose(np.sum(cond_prob_onehop[state_nr, k, :]), 1.0, abs_tol=1e-8), \
                                 "np.sum(cond_prob_onehop[state_nr=%d, k=%d])=%16.10f =? 1.0 =? %16.10f" \
                                 % (state_nr, k, np.sum(cond_prob_onehop[state_nr, k, :]), cumsum_condprob_onehop[state_nr, k])
                            
                            # The normalization can be fulfilled even though there are negative probabilities. 
                            #assert np.all(cond_prob_onehop[state_nr, k, :] > -LR.thresh)
                            if not np.all(cond_prob_onehop[state_nr, k, :] > -5e-8): # -0.00000
                                print("Error: Negative probabilities")
                                print("state_nr=", state_nr, "k=", k)
                                fh = open(self.dir+"NegativeProbabilities.dat", "a")
                                fh.write("state_nr=%d, k=%d\n" %(state_nr, k))
                                fh.write(" ".join([str(s) for s  in cond_prob_onehop[state_nr, k, :].flatten()]) + "\n")
                                fh.close()
                                print("ERROR: NEGATIVE PROBABILITIES - set cond. probs. equal to reference state")
                                print("CAREFUL ! THIS IS NOT JUSTIFIED")
                                cond_prob_onehop[state_nr, k, :] = cond_prob_ref[k, :]
                                #print("Exiting ...")                                
                                #exit(1)

        assert not np.any(np.isnan(cond_prob_onehop))

        return cond_prob_onehop.reshape(-1, self.N*self.D), cond_prob_ref.reshape((self.N*self.D,))


if __name__ == "__main__":

    import matplotlib.pyplot as plt 
    from test_suite import prepare_test_system_zeroT
    from slater_determinant import Slater2spOBDM
    
    from time import time 

    (Nsites, eigvecs) = prepare_test_system_zeroT(Nsites=200, potential='none', PBC=False, HF=False)
    for Np in (110, 120, 130, 140, 150, 160, 170, 180, 190, ):
        Nparticles = Np
        num_samples = 5

        SDsampler  = SlaterDetSampler_ordered(Nsites=Nsites, Nparticles=Nparticles, single_particle_eigfunc=eigvecs, naive_update=True)
        SDsampler1 = SlaterDetSampler_ordered(Nsites=Nsites, Nparticles=Nparticles, single_particle_eigfunc=eigvecs, naive_update=True)
        SDsampler2 = SlaterDetSampler_ordered(Nsites=Nsites, Nparticles=Nparticles, single_particle_eigfunc=eigvecs, naive_update=False)

    #    t0 = time()
    #    for _ in range(num_samples):
    #        occ_vec, _ = SDsampler1.sample()
    #    t1 = time()
    #    print("naive, elapsed=", (t1-t0)*20 )

        t0 = time()
        for _ in range(num_samples):
            occ_vec, _ = SDsampler2.sample()
        t1 = time()
        print("block update (cancel dets), elapsed=", (t1-t0) )

        ## Check that sampling the Slater determinant gives the correct average density. 
        #occ_vec = torch.zeros(Nsites)
        #for s in range(num_samples):
        #    occ_vec_, prob_sample = SDsampler2.sample()
        #    print("=================================================================")
        #    print("amp_sample= %16.8f"%(np.sqrt(prob_sample)))
        #    print("naive sampler: amplitude= %16.8f"%(SDsampler1.psi_amplitude(occ_vec_)))
        #    print("block update sampler: amplitude=", SDsampler2.psi_amplitude(occ_vec_))
        #    print("=================================================================")
        #    occ_vec += occ_vec_
           

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
