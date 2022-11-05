import numpy as np
from time import time 
import math 

from bitcoding import int2bin, bin2pos 

from k_copy import calc_k_copy
from monitoring_old import logger as mylogger
import lowrank_update as LR


def _detratio_from_scratch_v0(G, occ_vec, base_pos, i):
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


def _detratio_from_scratch_tmp(G, occ_vec, base_pos, i):
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
def _detratio_from_scratch(G, occ_vec, base_pos, i):
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
def lowrank_update_kinetic(GG, D, N, ref_I, xs_I, rs_pos, print_stats=True):
    """
    Probability density estimation on states connected to I_ref by the kinetic operator `kinetic_operator`,
    given the Slater-Jastrow ansatz. 
    
    This routine is similar to sampling, but not quite. It duplicates much of the sampling routine of the 
    Slater sampler. 

    Parameters:
    -----------
    GG : 2D numpy array
        Single-particle Green's function 
    D : int 
        Dimension of single-particle Hilbert space (i.e. number of sites)
    N : int 
        Number of particles
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
    xs = int2bin(xs_I, ns=D)
    ref_conf = int2bin(ref_I, ns=D)    
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

    cond_prob_ref = np.zeros((N, D))
    cond_prob_onehop = np.zeros((num_onehop_states, N, D))
    cumsum_condprob_onehop = np.zeros((num_onehop_states, N))

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

    for k in range(N):
        # Calculate the conditional probabilities for the k-th particle (for all onehop states 
        # connected to the reference state through the kinetic operator simultaneously, using a low-rank update).
        xmin = 0 if k==0 else pos_vec[k-1] + 1 # half-open interval (xmin included, xmax not included)
        xmax = D - N + k + 1
        Ksites = list(range(0, xmin))
        Ksites_add = Ksites.copy()

        Gnum_inv_reuse[k] = dict.fromkeys(range(xmin, xmax))

        if k >= 2:
            # don't waste memory
            Gnum_inv_reuse[k-2].clear()

        Gdenom = GG[np.ix_(Ksites, Ksites)] - np.diag(occ_vec[0:len(Ksites)])

        # In production runs use flag -O to suppress asserts and 
        # __debug__ sections. 
        if __debug__:
            if Gdenom.shape[0] > 0:
                cond = np.linalg.cond(Gdenom)
                if cond > mylogger.info_refstate.Gdenom_cond_max:
                    mylogger.info_refstate.Gdenom_cond_max = cond 
                    if cond > 1e5:
                        print("Gdenom_cond_max=", mylogger.info_refstate.Gdenom_cond_max)            

        det_Gdenom = np.linalg.det(Gdenom)

        # Internal state used during low-rank update of conditional probabilities 
        # of the connnecting states. 
        Gdenom_inv = np.linalg.inv(Gdenom)

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
            Gnum = GG[np.ix_(Ksites_add, Ksites_add)] - np.diag(occ_vec_add)
            
            det_Gnum = np.linalg.det(Gnum)

            # In case a cond. prob. of the reference state is zero:
            try:
                Gnum_inv = np.linalg.inv(Gnum)
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
                                        cond_prob_onehop[state_nr, k-1, i] = (-1) * _detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-2], i=i)                                             
                                else:
                                    print("Gnum_inv_reuse is None (i.e. accidental zero cond. prob. of reference state)")
                                    cond_prob_onehop[state_nr, k-1, i] = (-1) * _detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-2], i=i)                                                                             
                                assert -assert_margin <= cond_prob_onehop[state_nr, k-1, i] <= 1.0 + assert_margin, "cond prob = %16.10f" %(cond_prob_onehop[state_nr, k-1, i])


                            # additionally ...
                            ###if k == N-1 or k == k_s[state_nr]: 
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
                                            cond_prob_onehop[state_nr, k, i] = (-1) * _detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                        except LR.ErrorFinitePrecision as e:
                                            print("Excepting LR.ErrorFinitePrecision 1, state_nr=", state_nr, "k=", k, "i=", i, "msg=", e)
                                            cond_prob_onehop[state_nr, k, i] = (-1) * _detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                    else:
                                        print("Gnum_inv_reuse is None (i.e. accidental zero cond. prob. of reference state)")
                                        cond_prob_onehop[state_nr, k, i] = (-1) * _detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
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
                                    cond_prob_onehop[state_nr, k, i] = (-1) * _detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                except LR.ErrorFinitePrecision as e: #np.linalg.LinAlgError as e: # from inverting singular matrix in LR.adapt_Ainv() 
                                    print("Excepting LR.ErrorFinitePrecision 2, state_nr=", state_nr, "k=", k, "i=", i, "msg=", e)
                                    cond_prob_onehop[state_nr, k, i] = (-1) * _detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                assert -assert_margin <= cond_prob_onehop[state_nr, k, i] <= 1.0 + assert_margin, "cond_prob[state_nr=%d, k=%d, i=%d]=%16.10f" % (state_nr, k, i, cond_prob_onehop[state_nr, k, i])

                            # Yet another special case (only relevant in 2D)
                            elif k > k_s[state_nr] + 1 and i > s:
                                try:
                                    corr_factor = LR.removeadd_rs(Gnum_inv, Gdenom_inv, r, s)
                                    cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]                                                                                                                     
                                except LR.ErrorFinitePrecision as e: 
                                    print("Excepting finite precision error 2, state_nr=", state_nr, "k=", k, "i=", i, "msg=", e)
                                    cond_prob_onehop[state_nr, k, i] = (-1) * _detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
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
                                                        cond_prob_onehop[state_nr, k, j_add] = (-1) * _detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=j_add)
                                                else:
                                                    print("Gnum_inv_reuse is None (i.e. accidental zero cond. prob. of reference state)")
                                                    cond_prob_onehop[state_nr, k, j_add] = (-1) * _detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=j_add)
                                            except LR.ErrorFinitePrecision as e:
                                                    cond_prob_onehop[state_nr, k, j_add] = (-1) * _detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=j_add)                                                                
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
                                            cond_prob_onehop[state_nr, k, i] = (-1) * _detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                    except LR.ErrorFinitePrecision as e:
                                        cond_prob_onehop[state_nr, k, i] = (-1) * _detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
                                    assert -assert_margin <= cond_prob_onehop[state_nr, k, i] <= 1.0 + assert_margin, "cond prob = %16.10f" %(cond_prob_onehop[state_nr, k, i])

                                elif k > k_r[state_nr]: # conditional probs. of reference state and onehop state have the same support 
                                    try:                                           
                                        corr_factor = LR.removeadd_rs(Gnum_inv, Gdenom_inv, r, s)
                                        cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]
                                    except LR.ErrorFinitePrecision as e:
                                        print("k>k_r: Excepting finite precision error 3, state_nr=", state_nr, "k=", k, "i=", i, "msg=", e)
                                        print("ref_conf    =", ref_conf)
                                        print("xs[state_nr]=", xs[state_nr])                                            
                                        cond_prob_onehop[state_nr, k, i] = (-1) * _detratio_from_scratch(GG, occ_vec=xs[state_nr], base_pos=xs_pos[state_nr, k-1], i=i)
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
                    IN_SUPPORT = i in np.arange(xs_pos[state_nr, k-1] + 1, D - N + k + 1)
                    fh.write("%d %d %20.19f %s\n" % (k, i, cond_prob_onehop[state_nr, k, i], IN_SUPPORT))
            fh.close()   

    # Check that all conditional probabilities are normalized. 
    # IMPROVE: Return also information about violation of probability normalization. 
    if True: #__debug__:
            for state_nr, (k_copy_, (r,s)) in enumerate(onehop_info):
                for k in range(N):
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

    return cond_prob_onehop.reshape(-1, N*D), cond_prob_ref.reshape((N*D,))
