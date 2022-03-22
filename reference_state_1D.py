# TODO: - If the probability of the reference state is extremely low (1e-60), then 
#         there are numerical inaccuracies in the log_probs of the one-hop states. 
#         One could argue that in a VMC simulation this scenario should not occur.
#
#       - Add meaningful tests (in a different file).
#
#       - Correct cumsum. Problem occurs when probs at k are used to calculate probs at k-1.

import numpy as np
from time import time 

from test_suite import ( prepare_test_system_zeroT,
                         generate_random_config )
from bitcoding import *
from one_hot import *
from Slater_Jastrow_simple import kinetic_term, Lattice1d
from slater_sampler_ordered_memory_layout import SlaterDetSampler_ordered
from k_copy import *

from monitoring import logger

np.random.seed(43514)

# absolute tolerance in comparisons (e.g. normalization)
ATOL = 1e-8

# normalization needs to be satisfied up to 
#     \sum_i p(i)  > `eps_norm_probs``
eps_norm_probs = 1.0 - 1e-10


def log_cutoff(x):
    """
    Replace -inf by a very small, but finite value.
    """
    return np.where(x > 0, np.log(x), -1000)
 

def copy_cond_probs(cond_prob_ref, cond_prob_onehop, one_hop_info):
    """
    Copy all conditional probabilities for k <= k_copy_, which are identical in the reference 
    state and in the one-hop states.
    """
    for state_nr, (k_copy, _) in enumerate(one_hop_info):
        cond_prob_onehop[state_nr, 0:k_copy+1, :] = cond_prob_ref[0:k_copy+1, :]


def cond_prob2log_prob(xs, cond_probs_allk):
    """
    Pick conditional probabilities at actually sampled positions so as to 
    get the probability of the microconfiguration.

    Inputs are conditional probs.
    """
    xs = np.asarray(xs)
    xs_unfolded = occ_numbers_unfold(xs, duplicate_entries = False)
    xs_pos = bin2pos(xs)
    Np = len(xs_pos[0]); Ns = len(xs[0]); num_states = xs.shape[0]    
    mm = xs_unfolded * cond_probs_allk.reshape(-1, Np*Ns)
    # CAREFUL: this may be wrong if a probability is accidentally zero !
    # introduce a boolean array which indicates the valid support
    # and use log-probabilities throughout.  
    supp = np.empty((num_states, Np, Ns), dtype=bool)
    supp[...] = False 
    for l in range(num_states):
        for k in range(Np):
            xmin = 0 if k==0 else xs_pos[l, k-1] + 1
            xmax = Ns - Np + k + 1
            supp[l, k, xmin:xmax] = True
    supp = supp.reshape(-1, Np*Ns)
    assert mm.shape == supp.shape
    # CAREFUL
    log_probs = log_cutoff(np.where(mm > 0, mm, 1.0)).sum(axis=-1)
    #log_probs = log_cutoff(np.where(supp, mm, 1.0)).sum(axis=-1)
    return log_probs 


def cond_logprob2log_prob(xs, cond_logprobs_allk):
    """
    Pick conditional probabilities at actually sampled positions so as to 
    get the probability of the microconfiguration.

    Inputs are conditional log-probs.
    """
    xs = np.asarray(xs)
    xs_unfolded = occ_numbers_unfold(xs, duplicate_entries = False)
    xs_pos = bin2pos(xs)
    Np = len(xs_pos[0]); Ns = len(xs[0]); num_states = xs.shape[0]    
    mm = xs_unfolded * cond_logprobs_allk.reshape(-1, Np*Ns)
    # CAREFUL: this may be wrong if a probability is accidentally zero !
    # introduce a boolean array which indicates the valid support
    # and use log-probabilities throughout.  
    supp = np.empty((num_states, Np, Ns), dtype=bool)
    supp[...] = False 
    for l in range(num_states):
        for k in range(Np):
            xmin = 0 if k==0 else xs_pos[l, k-1] + 1
            xmax = Ns - Np + k + 1
            supp[l, k, xmin:xmax] = True
    supp = supp.reshape(-1, Np*Ns)
    assert mm.shape == supp.shape
    # CAREFUL
    log_probs = np.where(supp, mm, 0.0).sum(axis=-1)
    return log_probs 


# Calculate the conditional probabilities of the reference state
Ns = 16; Np = 7    # Ns=12, Np=5: singular matrix
l1d = Lattice1d(ns=Ns)
#l2d = Lattice_rectangular(nx=4, ny=4)
_, U = prepare_test_system_zeroT(Nsites=Ns, potential='none', Nparticles=Np)
P = U[:, 0:Np]
G = np.eye(Ns) - np.matmul(P, P.transpose(-1,-2))
SDsampler = SlaterDetSampler_ordered(Nsites=Ns, Nparticles=Np, single_particle_eigfunc=U, naive=False)

for jj in range(100):
    print("jj=", jj)

    ref_conf = generate_random_config(Ns, Np)
    ref_I = bin2int(ref_conf) # ATTENTION: Wrong results for too large bitarrays !
    print("ref_I=", ref_I)

    # # #  `states_I` are only the connecting states, the reference state is not included 
    rs_pos, states_I, _ = valid_states(*kinetic_term([ref_I], l1d))
    num_connecting_states = len(states_I)
    xs = int2bin(states_I, ns=Ns)

    print("ref")
    print(ref_conf)
    print("xs=")
    print(xs)


    #ref_conf = np.array([1, 0, 1, 1, 0, 0])
    #xs = list(         [[0, 0, 1, 1, 0, 1]],)
    #rs_pos = ((0, 5),)

    #ref_conf = np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    #xs = list(         [[1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0]],)
    #rs_pos = ((11, 0),)

    #ref_conf = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1])
    #xs = list(         [[1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]],)
    #rs_pos = ((11,0),)

    # ===============================================================
    # test case 1 
    #ref_conf = np.array( [1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1])
    #xs = list(          [[1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]],)
    #rs_pos = ((3, 8),)    

    # test case 2 
    #ref_conf = np.array( [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1])
    #xs = list(          [[1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1]],)
    #rs_pos = ((8, 3),)    

    # The conditional probs of ref_conf for test case 1 must be the same 
    # as the conditional probs. of onehop_state for test case 2. 
    # IMPROVE: Convert this into a doctest or create a test case in a
    # separate file. 
    # ===============================================================

    #ref_conf = np.array( [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    #xs = list(          [[0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1]],)
    #rs_pos = ((9, 3),)    

    #ref_conf= np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0])
    #xs = list(        [[0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]], )
    #rs_pos = ((0, 4), )

    #ref_conf= np.array([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1])
    #xs = list(        [[0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1]], )
    #rs_pos = ((0, 12),)

    #ref_conf = np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1])
    #xs = list(         [[0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1]], )
    #rs_pos = ((0, 4),)


    #ref_conf = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0])
    #xs = list(         [[1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]], )
    #rs_pos = ((6, 2),)

    #ref_conf = np.array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
    #xs = list(         [[1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1]], )
    #rs_pos = ((6, 2), )

    #ref_conf = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0])
    #xs = list(         [[0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]], )
    #rs_pos = ((13, 1), )


    num_connecting_states = len(xs)    

    # special case of 1d n.n. hopping matrix
    # assert np.all([abs(r-s) == 1 or abs(r-s) == Ns-1 for r,s in rs_pos])

    k_copy = calc_k_copy(rs_pos, ref_conf)
    print("k_copy=", k_copy)
    one_hop_info = list(zip(k_copy, rs_pos))


    # For s < r, the support for conditional probabilities in the connecting state 
    # is larger than in the reference state: xmin(conn) < xmin(ref). 
    # There the probabilities for additional sites need to be calculated which have no counterpart 
    # in the calculations for the reference state. 
    # 
    # S_connecting_states = ((s0, k0), (s1, k1), ...)
    #     ki-th particle sits at position si. 
    s_pos = list(s for (r,s) in rs_pos if s < r)
    S_connecting_states = list(zip(s_pos, [kk for idx, kk in enumerate(k_copy) if rs_pos[idx][1] < rs_pos[idx][0]]))
    det_Gnum_reuse = dict()

    cond_prob_ref = np.zeros((Np, Ns))
    cond_logprob_ref = np.zeros((Np, Ns))
    cond_prob_onehop = np.zeros((num_connecting_states, Np, Ns))
    cond_logprob_onehop = np.zeros((num_connecting_states, Np, Ns))
    cumsum_condprob_onehop = np.zeros((num_connecting_states, Np))

    # The following variables are needed for low-rank update for onehop states differing 
    # from the reference state by long-range hopping between positions r and s.
    Gdenom_inv_reuse = dict()
    det_Gdenom_reuse = dict()
    Gnum_inv_reuse = dict()


    Ksites = []
    occ_vec = list(ref_conf)
    assert type(occ_vec) == type(list()) # use a list, otherwise `occ_vec[0:xmin] + [1]` will result in `[]`. 
    pos_vec = bin2pos(ref_conf)

    xs = np.array(xs) # convert list of arrays into 2D array 
    xs_pos = bin2pos(xs)

    # Needed for long-range hopping in 2D. 
    # At position `s` (`r`) sits the `k_s[state_nr]`-th (`k_r[state_nr]`-th) particle. 
    # Here, k_s and k_r are counted in the the one-hop state. However, the loop index k 
    # is referring to the particle numbers in the reference state. 
    k_s = [np.searchsorted(xs_pos[state_nr, :], rs_pos[state_nr][1]) for state_nr in range(num_connecting_states)]
    k_r = [np.searchsorted(xs_pos[state_nr, :], rs_pos[state_nr][0]) for state_nr in range(num_connecting_states)]

    for k in range(Np):
        # Calculate the conditional probabilities for the k-th particle (for all connecting states 
        # simultaneously, using a low-rank update).
        xmin = 0 if k==0 else pos_vec[k-1] + 1 # half-open interval (xmin included, xmax not included)
        xmax = Ns - Np + k + 1
        Ksites = list(range(0, xmin))
        Ksites_add = Ksites.copy()

        Gnum_inv_reuse[k] = dict.fromkeys(range(xmin, xmax))

        if k >= 2:
            # don't waste memory
            Gnum_inv_reuse[k-2].clear()

        for ii, i in enumerate(range(xmin, xmax)):
            t0=time()
            # reference state        
            Ksites_add += [i]
            occ_vec_add = occ_vec[0:xmin] + [0]*ii + [1]
            Gnum = G[np.ix_(Ksites_add, Ksites_add)] - np.diag(occ_vec_add)
            Gdenom = G[np.ix_(Ksites, Ksites)] - np.diag(occ_vec[0:len(Ksites)])

            det_Gnum = np.linalg.det(Gnum)
            det_Gdenom = np.linalg.det(Gdenom)

            # Internal state used during low-rank update of conditional probabilities 
            # of the connnecting states. 

            # In case a cond. prob. of the reference state is zero:
            try:    
                Gnum_inv = np.linalg.inv(Gnum) 
            except np.linalg.LinAlgError as e:
                print("ERROR: det_Gnum=%16.12f\n" % (det_Gnum), e)
                exit(1)
            Gdenom_inv = np.linalg.inv(Gdenom)

            # Needed for low-rank update for onehop states differing from the reference 
            # state by long-range hopping between positions r and s. 
            Gdenom_inv_reuse[k] = Gdenom_inv # does not depend on i 
            det_Gdenom_reuse[k] = det_Gdenom # does not depend on i
            Gnum_inv_reuse[k][i] = Gnum_inv

            cond_prob_ref[k, i] = (-1) * det_Gnum / det_Gdenom
            cond_logprob_ref[k,i] = log_cutoff(abs(det_Gnum)) - log_cutoff(abs(det_Gdenom))
            t1 = time() 
            logger.info_refstate.elapsed_ref += (t1 - t0)

            if (i,k) in S_connecting_states:
                det_Gnum_reuse.update({k : det_Gnum})

            # Now calculate the conditional probabilities for all states related 
            # to the reference state by one hop, using a low-rank update of `Gnum`
            # and 
            # `Gdenom`.
            t0_conn = time()
            for state_nr, (k_copy_, (r,s)) in enumerate(one_hop_info):
                if k_copy_ >= k:
                    # Copy conditional probabilities rather than calculating them.
                    # Exit the loop; it is assumed that one-hop states are ordered according to increasing values 
                    # of k_copy_, i.e. this condition is also fulfilled for all subsequent one-hop states.
                    break 
                else: # k_copy < k  
                    # SOME SPECIAL CASES 
                    xmin_onehop = xs_pos[state_nr, k-1] + 1; xmax_onehop = xmax 
                    if xmin_onehop == xmax_onehop-1 and r < i and s < i: 
                    # Every subsequent empty site needs to be occupied to accomodate all particles both in the reference state 
                    # and in the one-hop state. 
                        cond_prob_onehop[state_nr, k, i] = 1.0
                        cumsum_condprob_onehop[state_nr, k] = 1.0
                        continue

                    if abs(r-s) > 1: # long-range hopping in 1d
                        if r < s:
                            if k > 1 and k <= k_s[state_nr]: # k=0 can always be copied from the reference state 
                                corr_factor_Gnum = corr_factor_remove_r(Gnum_inv_reuse[k][i], r=r)
                                corr_factor_Gdenom = corr_factor_remove_r(Gdenom_inv_reuse[k], r=r)
                                corr_factor = corr_factor_Gnum / corr_factor_Gdenom
                                # NOTE: The cond. probs. for (k-1)-th particle are computed retroactively while the cond. probs. for k-th particle 
                                # of the reference state are being computed. 
                                cond_prob_onehop[state_nr, k-1, i] = corr_factor * cond_prob_ref[k, i]

                            # additionally ...
                            if k == Np-1 or k == k_s[state_nr]: 
                                          # Special case: cond. probs. for last particle (for 1D system with pbc) or 
                                          # last particle whose support involves position `s` (for 2D system).
                                          # IMPROVE: adapt to 2D long-range hoppping 
                                          # where more particles come after positions s.           
                                if i > xs_pos[state_nr, k-1]: # support is smaller than in the reference state 
                                    
                                    # In this special case, calculate the ratio of determinants from scratch.
                                    base = list(range(0, xs_pos[state_nr, k-1]+1))
                                    occ_vec_base = list(xs[state_nr, 0:xs_pos[state_nr, k-1]+1])
                                    Gdenom_special = G[np.ix_(base, base)] - np.diag(occ_vec_base)
                                    extend = list(range(0, i+1))
                                    occ_vec_add = [1] if (i - xs_pos[state_nr, k-1] == 1) else [0] * (i - xs_pos[state_nr, k-1] - 1) + [1]
                                    occ_vec_extend = occ_vec_base + occ_vec_add
                                    Gnum_special = G[np.ix_(extend, extend)] - np.diag(occ_vec_extend)

                                    cond_prob_onehop[state_nr, k, i] = (-1) * np.linalg.det(Gnum_special) / np.linalg.det(Gdenom_special)
                                    print("special=", cond_prob_onehop[state_nr, k, i])

                                    # !!!!!!!!!!! Problem: np.linalg.inv() sometimes throws error due to singular matrix 
                                    # try:
                                    #     Gnum_inv_, corr1 = adapt_Ainv(Gnum_inv_reuse[k][xs_pos[state_nr, k-1]], Gglobal=G, r=r, s=s, i_start=xs_pos[state_nr, k-1]+1, i_end=i)                                       
                                    # except np.linalg.LinAlgError as e :
                                    #     Gnum_inv_2, corr1 = adapt_Ainv_sing(Gdenom_inv_reuse[k], Gglobal=G, r=r, s=s, i_start=xs_pos[state_nr, k-2]+1, i_end=i, pos1=xs_pos[state_nr, k-1], pos2=i)
                                    # corr2 = corr_factor_remove_r(Gnum_inv_, r=r)                                               

                                    # corr_factor_Gnum = corr1 * corr2               
                                    # print("corr_factor_Gnum=", corr_factor_Gnum)                      
                                    # Gdenom_inv_ = Gnum_inv_reuse[k][xs_pos[state_nr, k-1]]
                                    # corr_factor_Gdenom = corr_factor_remove_r(Gdenom_inv_, r=r) * ( det_Gnum / det_Gdenom )
                                    # corr_factor = corr_factor_Gnum / corr_factor_Gdenom 
                                    # cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]
                            
                            # Yet another special case 
                            if k == k_s[state_nr] + 1 and i > s:
                                # Again, calculate the ratio of determinants from scratch. 
                                # (same code as above, combine it with the previous case)
                                base = list(range(0, xs_pos[state_nr, k-1]+1))
                                occ_vec_base = list(xs[state_nr, 0:xs_pos[state_nr, k-1]+1])
                                Gdenom_special = G[np.ix_(base, base)] - np.diag(occ_vec_base)
                                extend = list(range(0, i+1))
                                occ_vec_add = [1] if (i - xs_pos[state_nr, k-1] == 1) else [0] * (i - xs_pos[state_nr, k-1] - 1) + [1]
                                occ_vec_extend = occ_vec_base + occ_vec_add
                                Gnum_special = G[np.ix_(extend, extend)] - np.diag(occ_vec_extend)

                                cond_prob_onehop[state_nr, k, i] = (-1) * np.linalg.det(Gnum_special) / np.linalg.det(Gdenom_special)


                            if k > k_s[state_nr] + 1 and i > s:
                                corr_factor_Gnum = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s)
                                corr_factor_Gdenom = corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s)
                                corr_factor = corr_factor_Gnum / corr_factor_Gdenom
                                cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]


                        elif r > s:
                            if i > s:
                                if k <= k_r[state_nr]:
                                    if i==pos_vec[k-1]+1:
                                    # need to calculate some additional cond. probs. along the way 
                                        for j_add in range(xmin_onehop, pos_vec[k-1]+1):
                                            print("state_nr=", state_nr, "s=", s, "k=", k, "i=", i, "j_add=", j_add, "xmin_onehop=", xmin_onehop, "k_copy_=", k_copy_)
                                            # reuse all information from (k-1) cond. probs. of reference state
                                            if k == k_copy_ + 1 and s == 0: # 1D long-range hopping                                             
                                                Gdenom_inv_, corr = adapt_Gdenom_inv(Gdenom_inv_reuse[k-1], Gglobal=G, r=r, s=xs_pos[state_nr, k-1])
                                                corr_factor_Gdenom = corr
                                            elif k == k_copy_ + 1 and s > 0: # 2D long-range hopping: There are particles or empty sites to the left of s.                                            
                                                # 1. No particle to the left of position `s`. (Of course, this is so both in the reference state and in the onhop
                                                # state since they differ only in the occupancies of the positions `s` and `r`.)
                                                if k_s[state_nr] == 0:
                                                    print("special case 1")
                                                    Gdenom_inv_, corr = adapt_Ainv(np.array([]).reshape(0,0), Gglobal=G, r=r, s=s, i_start=0, i_end=s)
                                                    corr_factor_Gdenom = corr
                                                    #print("exiting...")
                                                    #exit(1)
                                                # 2. There is at least one particle to the left of position `s`. Numerator and denominator 
                                                # matrices can be extended from that position. 
                                                elif k_s[state_nr] > 0:
                                                    print("special case 2")
                                                    print("xs_pos[state_nr,:]=", xs_pos[state_nr,:])
                                                    print("pos_vec[:]        =", pos_vec[:])
                                                    # We are calculating cond. probs. for the k-th particle. For the reference state low-rank updates 
                                                    # are based on the sampled position of the (k-1)-th particle. For the onehop state they are based 
                                                    # on the position of the (k-2)-th particle. 
                                                    Gdenom_inv_, corr = adapt_Ainv(Gdenom_inv_reuse[k-1], Gglobal=G, r=r, s=s, i_start=pos_vec[k-2]+1, i_end=s)
                                                    corr_factor_Gdenom = corr
                                                    #print("exiting ...")
                                                    #exit(1)
                                            else:
                                                corr_factor_Gdenom = corr_factor_add_s(Gdenom_inv_reuse[k-1], s=s)
                                            corr_factor_Gnum = corr_factor_add_s(Gnum_inv_reuse[k-1][j_add], s=s)
                                            corr_factor = corr_factor_Gnum / corr_factor_Gdenom
                                            cond_prob_onehop[state_nr, k, j_add] = corr_factor * cond_prob_ref[k-1, j_add]
                                            # update cumul. probs. explicitly because this is inside the body of an extra loop
                                            cumsum_condprob_onehop[state_nr, k] += cond_prob_onehop[state_nr, k, j_add]

                                    if k == k_copy_ + 1 and s==0:                                                                                       
                                        Gdenom_inv_, corr_factor_Gdenom = adapt_Gdenom_inv(Gdenom_inv_reuse[k-1], Gglobal=G, r=r, s=s)
                                    elif k == k_copy_ + 1 and s > 0:
                                        #print("DOING NOTHING, i=", i)
                                        # IS THIS CORRECT ? WHAT ABOUT i < r ? 
                                        print("Is this correct 1 , r=", r, "i=", i)
                                        Gdenom_inv_, corr_factor_Gdenom = adapt_Gdenom_inv(Gdenom_inv_reuse[k-1], Gglobal=G, r=r, s=s)
                                        #pass # Gdenom_inv_ and corr_factor_Gdenom has already been computed and is still up to date 
                                    else:                 
                                        corr_factor_Gdenom= corr_factor_add_s(Gdenom_inv_reuse[k-1], s=s)
                                    print("Is this correct 2 , r=", r, "i=", i, "No, No, No !")  # NO !                             
                                    corr_factor_Gnum = corr_factor_removeadd_rs(Gnum_inv, r=pos_vec[k-1], s=s)                                    
                                    corr_factor = corr_factor_Gnum / corr_factor_Gdenom
                                    cond_prob_onehop[state_nr, k, i] = corr_factor * (det_Gdenom / det_Gdenom_reuse[k-1]) * cond_prob_ref[k, i]

                                elif k > k_r[state_nr]:
                                    corr_factor_Gnum = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s)
                                    corr_factor_Gdenom = corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s)
                                    corr_factor = corr_factor_Gnum / corr_factor_Gdenom
                                    cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]
                                
                    elif abs(r-s) == 1: # 1d nearest neighbour hopping 
                        pass
                        if r < s:
                            if i > s:
                                if not np.isclose(det_Gnum, 0.0, atol=1e-16): # don't invert a singular matrix 
                                    logger.info_refstate.counter_nonsingular += 1
                                    cond = np.linalg.cond(Gdenom)
                                    if cond > logger.info_refstate.Gdenom_cond_max:
                                        logger.info_refstate.Gdenom_cond_max = cond 
                                        print("Gdenom_cond_max=", logger.info_refstate.Gdenom_cond_max)

                                    if k==(k_copy_+1):                                                   
                                        Gdenom_inv_, corr1 = adapt_Gdenom_inv(Gdenom_inv, Gglobal=G, r=r, s=s)
                                        corr2 = corr_factor_remove_r(Gdenom_inv_, r=r)
                                        corr_factor_Gdenom = corr1 * corr2
                                        corr_factor_Gnum = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s)
                                        corr_factor = corr_factor_Gnum / corr_factor_Gdenom 
                                        log_corr_factor = log_cutoff(abs(corr_factor_Gnum)) - log_cutoff(abs(corr1)) - log_cutoff(abs(corr2))                                                                      
                                    else:
                                        # Correction factor in the numerator is a problem if Gnum_inv[r,r] \approx 1. 
                                        corr_factor = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s) \
                                                        / corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s)
                                        log_corr_factor = ( log_cutoff(abs(corr_factor_removeadd_rs(Gnum_inv, r=r, s=s)))
                                                        - log_cutoff(abs(corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s))) )
                                    if corr_factor < 0 and -corr_factor < 1e-4: corr_factor = 0.0
                                    assert corr_factor >= 0, "state_nr=%d, k=%d, i=%i, corr_factor=%16.15f" % (state_nr, k, i, corr_factor)    
                                    cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i] 
                                    cond_logprob_onehop[state_nr, k, i] = cond_logprob_ref[k, i] + log_corr_factor 

                                else: 
                                    # As the numerator is singular, the conditional probabilities of the connecting states 
                                    # should be calculated based on the matrix in the denominator, the inverse and determinant 
                                    # of which are assumed to be known. The matrix in the denominator cannot be singular. 
                                    t0 = time()
                                    # # First check whether the conditional probabilities are already saturated.
                                    # if np.isclose(sum(cond_prob_onehop[state_nr, k, xmin:i-1]), 1.0):  # CHECK: Why i-1 ? 
                                    #     cond_prob_onehop[state_nr, k, i-1:] = 0.0
                                    #     break
                                    # NOTE: The cond. prob. at the actually sampled positions needs to be computed before 
                                    #       saturation of the normalization can be exploited. 
                                    if cumsum_condprob_onehop[state_nr, k] > eps_norm_probs and i > xs_pos[state_nr, k]:  
                                        cond_prob_onehop[state_nr, k, i-1:] = 0.0
                                        logger.info_refstate.counter_skip += (xmax - i) 
                                        continue

                                    # print("numerator 1 singular, k=", k, " i=", i, "sum(cond_prob_ref)=", sum(cond_prob_ref[k, 0:i+1]), sum(cond_prob_onehop[state_nr, k, 0:i]))
                                    if k==(k_copy_+1):
                                        Gdenom_ = adapt_Gdenom(Gnum, r=r, s=s)
                                        Gdenom_inv_, corr1 = adapt_Gdenom_inv(Gdenom_inv, Gglobal=G, r=r, s=s)
                                        # Now Gdenom_inv_ still has a particle at position s and (!) r. 
                                        corr2 = corr_factor_remove_r(Gdenom_inv_, r=r)
                                        corr_factor_Gdenom = corr1 * corr2                                
                                        #assert np.isclose(np.linalg.det(Gdenom_), corr_factor_Gdenom * det_Gdenom)

                                        #Gnum_ = Gnum_from_Gdenom3(Gdenom_, Gglobal=G, r=r, s=s, i=i)
                                        corr4, corr3 = corr3_Gnum_from_Gdenom(Gdenom_inv_, Gglobal=G, r=r, s=s, xmin=xmin, i=i)
                                        det_Gnum_ = corr4 * corr3 * corr1
                                        #assert np.isclose(det_Gnum_, np.linalg.det(Gnum_))
                                        #cond_prob_onehop[state_nr, k, i] = (-1) * np.linalg.det(Gnum_) / np.linalg.det(Gdenom_)   
                                        cond_prob_onehop[state_nr, k, i] = (-1) * det_Gnum_ / (corr_factor_Gdenom) 
                                        #print("cond_prob_onehop[state_nr, k, i] = ", cond_prob_onehop[state_nr, k, i], np.linalg.det(Gnum_))
                                        cond_logprob_onehop[state_nr, k, i] = ( log_cutoff(abs(corr4)) + log_cutoff(abs(corr3)) + log_cutoff(abs(corr1)) 
                                                                                - log_cutoff(abs(corr_factor_Gdenom)) )

                                    else:
                                        # connecting state and reference state have the same support in the denominator 
                                        ### Gdenom_inv = np.linalg.inv(Gdenom) OK
                                        corr_factor_Gdenom = corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s)
                                        #Gnum_ = adapt_singular_Gnum(Gnum, r=r, s=s, i=i) # not useful  
                                        det_Gnum_ = det_Gnum_from_Gdenom(Gdenom_inv, det_Gdenom, Gglobal=G, r=r, s=s, xmin=xmin, i=i)
                                        #assert np.isclose(det_Gnum_, np.linalg.det(Gnum_))
                                        cond_prob_onehop[state_nr, k, i] = (-1) * det_Gnum_ / (det_Gdenom * corr_factor_Gdenom)
                                        #cond_prob_onehop[state_nr, k, i] = (-1) * np.linalg.det(Gnum_) / (det_Gdenom * corr_factor_Gdenom)
                                        cond_logprob_onehop[state_nr, k, i] = log_cutoff(abs(det_Gnum_)) - log_cutoff(abs(det_Gdenom)) - log_cutoff(abs(corr_factor_Gdenom))

                                    logger.info_refstate.counter_singular += 1
                                    #print("SSingular, detGnum=", det_Gnum, "detGdenom=", det_Gdenom, "cond_prob_ref[k, i]=", cond_prob_ref[k, i])
                                    #print("det_Gnum_=", det_Gnum_, "corr_factor_Gdenom=", corr_factor_Gdenom)

                                    t1 = time()
                                    logger.info_refstate.elapsed_singular += (t1 - t0)
                        elif r > s: 
                                # The support is larger than in the reference state. One needs to calculate (r-s)
                                # more conditional probabilities than in the reference state. 
                                # In other words, here,  i not in (xmin, xmax). 
                                # The case i == r is special. 

                                if not np.isclose(det_Gnum, 0.0, atol=1e-16): # don't invert a singular matrix                                                  
                                    logger.info_refstate.counter_nonsingular += 1

                                    # cond = np.linalg.cond(Gdenom)
                                    # if cond > Gdenom_cond_max:
                                    #     Gdenom_cond_max = cond 

                                    if k==(k_copy_+1):
                                        det_Gdenom_ = det_Gnum_reuse.get(k-1)
                                        if i==(r+1):
                                            # Actually we wish to calculate i==r, but since such an i is not in (xmin, xmax),
                                            # it will never appear in the iteration. Therefore this case is treated explicitly 
                                            # here. Since this case does not appear for the reference state, a "correction factor"
                                            # is not calculated, instead the cond. prob. is calculated directly:
                                            det_Gnum_ = det_Gdenom * corr_factor_add_s(Gdenom_inv, s=s)
                                            cond_prob_onehop[state_nr, k, i-1] = (-1) * det_Gnum_ / det_Gdenom_
                                            cond_logprob_onehop[state_nr, k, i-1] = log_cutoff(abs(det_Gnum_)) - log_cutoff(abs(det_Gdenom_))
                                            cumsum_condprob_onehop[state_nr, k] += cond_prob_onehop[state_nr, k, i-1]
                                        if i > r:  
                                            corr_factor1 = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s)
                                            corr_factor  = corr_factor1 * (det_Gdenom / det_Gdenom_)
                                            cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]
                                            cond_logprob_onehop[state_nr, k, i] = ( log_cutoff(abs(corr_factor1)) + log_cutoff(abs(det_Gdenom)) 
                                                                                    - log_cutoff(abs(det_Gdenom_)) + cond_logprob_ref[k, i] )
        
                                    else:
                                        corr_factor = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s) \
                                                        / corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s)     
                                        log_corr_factor = ( log_cutoff(abs(corr_factor_removeadd_rs(Gnum_inv, r=r, s=s))) 
                                                        - log_cutoff(abs(corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s)))                                                    )
                                        if corr_factor < 0 and -corr_factor < 1e-4: corr_factor = 0.0
                                        assert corr_factor >= 0, "state_nr=%d, k=%d, i=%i, corr_factor=%16.15f" % (state_nr, k, i, corr_factor)    
                                        cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]     
                                        cond_logprob_onehop[state_nr, k, i] = log_corr_factor + cond_logprob_ref[k, i]

                                else:
                                    # As the numerator is singular, the conditional probabilities of the connecting states 
                                    # should be calculated based on the matrix in the denominator, the inverse and determinant 
                                    # of which are assumed to be known. The matrix in the denominator cannot be singular.                             
                                    t0 = time()
                                    # # First check whether the conditional probabilities are already saturated.
                                    # if np.isclose(sum(cond_prob_onehop[state_nr, k, xmin:i-1]), 1.0): # CHECK: Why i-1 ? 
                                    #     cond_prob_onehop[state_nr, k, i-1:] = 0.0
                                    #     break        
                                    # First check whether the conditional probabilities are already saturated.
                                    # NOTE: The cond. prob. at the actually sampled positions needs to be computed before 
                                    #       saturation of the normalization can be exploited.                                 
                                    if cumsum_condprob_onehop[state_nr, k] > eps_norm_probs and i > xs_pos[state_nr, k]:
                                        cond_prob_onehop[state_nr, k, i-1:] = 0.0
                                        logger.info_refstate.counter_skip += (xmax - i)
                                        continue                                    
                                    logger.info_refstate.counter_singular += 1
                                    #print("SSingular")         
                                    # print("numerator 2 singular, k=", k, " i=", i, "sum(cond_prob)=", sum(cond_prob_ref[k, 0:i+1]))

                                    if k==(k_copy_+1):
                                        det_Gdenom_ = det_Gnum_reuse.get(k-1)
                                        if i==(r+1):
                                            # Actually we wish to calculate i==r, but since such an i is not in (xmin, xmax),
                                            # it will never appear in the iteration. Therefore this case is treated explicitly 
                                            # here. Since this case does not appear for the reference state, a "correction factor"
                                            # is not calculated, instead the cond. prob. is calculated directly:
                                            
                                            #print("singular numerator, i==r+1")
                                            #Gnum_ = Gnum_from_Gdenom_ieqrp1(Gdenom, r=r, s=s, i=i)           
                                            det_Gnum_ = det_Gdenom * corr_factor_add_s(Gdenom_inv, s=s)
                                            #assert np.isclose(det_Gnum_, np.linalg.det(Gnum_))
                                            #print("passed assert")                                    
                                            cond_prob_onehop[state_nr, k, i-1] = (-1) * det_Gnum_ / det_Gdenom_
                                            cond_logprob_onehop[state_nr, k, i-1] = log_cutoff(abs(det_Gdenom)) + log_cutoff(abs(corr_factor_add_s(Gdenom_inv, s=s))) - log_cutoff(abs(det_Gdenom_))

                                            cumsum_condprob_onehop[state_nr, k] += cond_prob_onehop[state_nr, k, i-1]
                                            #cond_prob_onehop[state_nr, k, i-1] = (-1) * np.linalg.det(Gnum_) / det_Gdenom_
                                        if i > r:                                     
                                            #Gnum_ = adapt_singular_Gnum2(Gnum, r=r, s=s, i=i)
                                            #print("Gnum_.shape=", Gnum_.shape)
                                            det_Gnum_ = det_Gnum_from_Gdenom(Gdenom_inv, det_Gdenom, Gglobal=G, r=r, s=s, xmin=xmin, i=i)
                                            ##print("det_Gnum_=", det_Gnum_, "np.linalg.det(Gnum_)=", np.linalg.det(Gnum_))
                                            #assert np.isclose(det_Gnum_, np.linalg.det(Gnum_) )
                                            cond_prob_onehop[state_nr, k, i] = (-1) * det_Gnum_ / det_Gdenom_
                                            cond_logprob_onehop[state_nr, k, i] = log_cutoff(abs(det_Gnum_)) - log_cutoff(abs(det_Gdenom_))

                                            #cond_prob_onehop[state_nr, k, i] = (-1) * np.linalg.det(Gnum_) / det_Gdenom_  
                                    else:
                                        #print("singular numerator, k != k_copy+1, k=", k, "i=", i)
                                        corr_factor_Gdenom = corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s) # OK
                                        #Gnum_ = adapt_singular_Gnum2(Gnum, r=r, s=s, i=i) 
                                        det_Gnum_ = det_Gnum_from_Gdenom(Gdenom_inv, det_Gdenom, Gglobal=G, r=r, s=s, xmin=xmin, i=i)
                                        #print("det_Gnum_=", det_Gnum_, "np.linalg.det(Gnum_)=", np.linalg.det(Gnum_))  
                                        cond_prob_onehop[state_nr, k, i] = (-1) * det_Gnum_ / ( det_Gdenom * corr_factor_Gdenom)      
                                        cond_logprob_onehop[state_nr, k, i] = log_cutoff(abs(det_Gnum_)) - log_cutoff(abs(det_Gdenom)) - log_cutoff(abs(corr_factor_Gdenom))
                                        #cond_prob_onehop[state_nr, k, i] = (-1) * np.linalg.det(Gnum_) / ( det_Gdenom * corr_factor_Gdenom)
                                        
                                    t1 = time()
                                    logger.info_refstate.elapsed_singular += (t1 - t0)

                    cumsum_condprob_onehop[state_nr, k] += cond_prob_onehop[state_nr, k, i]                    

            t1_conn = time()
            logger.info_refstate.elapsed_connecting_states += (t1_conn - t0_conn)                    


    fh = open("cond_prob_ref.dat", "w")
    fh.write("# ref_state ["+" ".join(str(item) for item in ref_conf)+"]\n\n")
    for k in range(cond_prob_ref.shape[0]):
        for i in range(cond_prob_ref.shape[1]):
            fh.write("%d %d %e\n" % (k, i, cond_prob_ref[k, i]))
    fh.close()

    copy_cond_probs(cond_prob_ref, cond_prob_onehop, one_hop_info)
    copy_cond_probs(cond_logprob_ref, cond_logprob_onehop, one_hop_info)

    for state_nr in range(cond_prob_onehop.shape[0]):
        fh = open("cond_prob_onehop%d.dat" % (state_nr), "w")
        fh.write("# ref_state ["+" ".join(str(item) for item in ref_conf)+"]\n")
        fh.write("# 1hop      ["+" ".join(str(item) for item in xs[state_nr])+"]\n")
        for k in range(cond_prob_onehop.shape[1]):
            for i in range(cond_prob_onehop.shape[2]):
                fh.write("%d %d %20.19f\n" % (k, i, cond_prob_onehop[state_nr, k, i]))
        fh.close()    

    log_probs = cond_prob2log_prob(xs, cond_prob_onehop)
    log_probs2 = cond_logprob2log_prob(xs, cond_logprob_onehop)
    log_prob_ref = cond_prob2log_prob([ref_conf], cond_prob_ref[None,:])


    for state_nr, (k_copy_, (r,s)) in enumerate(one_hop_info):
        for k in range(Np):
            if k > k_copy_:
                # pass
                print("=============================================")
                print("k=", k, "state_nr=", state_nr, "cumul=", cumsum_condprob_onehop[state_nr,k])
                print("ref_conf=", ref_conf)
                print("1hop sta=", xs[state_nr])
                print("k_r=", k_r[state_nr], "k_s=", k_s[state_nr])
                print("cond_prob_onehop[state_nr, k, :]=", cond_prob_onehop[state_nr, k, :])
                print("sum [state_nr=%d, k=%d]= %16.14f" % (state_nr, k, np.sum(cond_prob_onehop[state_nr, k, :])))
                print("cumsum_condprob_onehop[state_nr=%d, k=%d]=%16.14f" % (state_nr, k, cumsum_condprob_onehop[state_nr, k]) )
                assert np.isclose(np.sum(cond_prob_onehop[state_nr, k, :]), 1.0, atol=ATOL), "np.sum(cond_prob_onehop[state_nr=%d, k=%d])=%16.10f ?= %16.10f = cumsum" % (state_nr, k, np.sum(cond_prob_onehop[state_nr, k, :]), cumsum_condprob_onehop[state_nr, k])               


    for i in range(num_connecting_states):
        print("state_nr=", i)
        print("log_prob_ref=", log_prob_ref)
        print("ref_conf=", ref_conf)
        print("xs=      ", xs[i])
        print(log_probs[i], log_probs2[i], SDsampler.log_prob([xs[i]]).item())
        print("ratio=", np.exp(log_prob_ref - log_probs2[i]), xs[i], ref_conf)
        #assert np.isclose( log_probs2[i] - log_prob_ref, SDsampler.log_prob([xs[i]]).item()  - log_prob_ref, atol=1e-1)


    logger.info_refstate.print_summary()


def _test():
    import doctest
    doctest.testmod(verbose=False)

if __name__ == "__main__":
    _test()
