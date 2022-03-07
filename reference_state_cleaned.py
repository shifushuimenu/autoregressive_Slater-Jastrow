# TODO:

import numpy as np
from test_suite import prepare_test_system_zeroT
from bitcoding import *
from one_hot import *
from Slater_Jastrow_simple import kinetic_term, Lattice1d
from slater_sampler_ordered_memory_layout import SlaterDetSampler_ordered


from k_copy import *

from time import time 

np.random.seed(42994)
ATOL = 1e-8



def copy_cond_probs(cond_prob_ref, cond_prob_onehop, one_hop_info):
    """copy all conditional probabilities which are identical in the reference 
    state and in the one-hop states"""
    for state_nr, (k_copy, _) in enumerate(one_hop_info):
        print("k_copy=", k_copy)
        cond_prob_onehop[state_nr, 0:k_copy+1, :] = cond_prob_ref[0:k_copy+1, :]


def cond_prob2log_prob(xs, cond_probs_allk):
    """pick conditional probabilities at actually sampled positions so as to 
    get the probability of the microconfiguration"""
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
    log_probs = np.log(np.where(mm > 0, mm, 1.0)).sum(axis=-1)
    #log_probs = np.log(np.where(supp, mm, 1.0)).sum(axis=-1)
    return log_probs 


def Gnum_from_Gdenom3(Gdenom_, Gglobal, r, s, i):
    assert s > r 
    assert i > r
    #assert Gdenom.shape == (r+1, r+1)
    Ablock = Gdenom_[np.ix_(list(range(0, s+1)), list(range(0, s+1)))]
    Bblock = Gglobal[np.ix_(list(range(0, s+1)), list(range(s+1, i+1)))]
    Cblock = Bblock.transpose()
    Dblock = Gglobal[np.ix_(list(range(s+1, i+1)), list(range(s+1, i+1)))]
    G = np.block([[Ablock, Bblock],[Cblock, Dblock]])
    G[i,i] = G[i,i] - 1 
    return G

# Calculate the conditional probabilities of the reference state
Ns = 24; Np = 12    # Ns=20, Np=10: normlization problems with some cond. probs.  
_, U = prepare_test_system_zeroT(Nsites=Ns, potential='none', Nparticles=Np)
P = U[:, 0:Np]
G = np.eye(Ns) - np.matmul(P, P.transpose(-1,-2))


SDsampler = SlaterDetSampler_ordered(Nsites=Ns, Nparticles=Np, single_particle_eigfunc=U, naive=False)


eps_norm_probs = 1.0 - 1e-8 # Note: np.isclose(1.0 - 1e-5, 1.0) == True

def gen_random_config(Ns, Np):
    """generate a random reference state of fixed particle number"""
    config = np.zeros((Ns,), dtype=int) 
    config[0] = 1; config[-1] = 1 # !!!!! REMOVE: Make sure no hopping across p.b.c can occur. 
    counter = 0
    while counter < Np-2:
        pos = np.random.randint(low=0, high=Ns, size=1)
        if config[pos] != 1:
            config[pos] = 1
            counter += 1 
    return config


for jj in range(1):
    print("jj=", jj)

    ref_conf = gen_random_config(Ns, Np)
    ref_I = bin2int(ref_conf).numpy() # ATTENTION: Wrong results for too large bitarrays !

    l1d = Lattice1d(ns=Ns)
    # `states_I` are only the connecting states, the reference state is not included 
    # rs_pos, states_I, _ = valid_states(*kinetic_term([ref_I], l1d))
    # num_connecting_states = len(states_I)
    # xs = int2bin(states_I, ns=Ns)

    # JUST FOR TESTING THE SCALING 
    # Find all connecting states 
    xs = []
    rs_pos = []
    for r in range(len(ref_conf)):
        if r < len(ref_conf)-1 and r > 0:
            for s in (r - 1, r + 1):
                if ref_conf[r] == 1 and ref_conf[s] == 0:
                    x_temp = ref_conf.copy()
                    x_temp[r] = 0; x_temp[s] = 1
                    xs.append(x_temp)
                    rs_pos.append((r,s))

    num_connecting_states = len(xs)

    #print(ref_conf)
    #for i in range(len(xs)):
    #   print(xs[i])
    # END: JUST FOR TESTING THE SCALING 


    # ref_conf = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1])
    # xs = list([[1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]],)
    # num_connecting_states = len(xs)
    # rs_pos = ((2,3),)

    # special case of 1d n.n. hopping matrix 
    assert np.all([abs(r-s) == 1 or abs(r-s) == Ns-1 for r,s in rs_pos])
    k_copy = calc_k_copy(rs_pos, ref_conf)
    #print("k_copy=", k_copy)
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
    cond_prob_onehop = np.zeros((num_connecting_states, Np, Ns))
    cumul_sum_cond_prob_onehop = np.zeros((num_connecting_states, Np))

    Ksites = []
    occ_vec = list(ref_conf)
    assert type(occ_vec) == type(list()) # use a list, otherwise `occ_vec[0:xmin] + [1]` will result in `[]`. 
    pos_vec = bin2pos(ref_conf)
    xs = np.array(xs) # convert list of arrays into 2D array 
    xs_pos = bin2pos(xs)
    print("xs_pos=", xs_pos)


    elapsed_ref = 0.0
    elapsed_connecting_states = 0.0
    elapsed_singular = 0.0
    elapsed_adapt = 0.0
    counter_nonsingular = 0
    counter_singular = 0
    counter_skip = 0
    counter_refskip = 0
    det_Gdenom_array = np.zeros((Np, Ns))

    # REMOVE
    Gdenom_cond_max = 0

    for k in range(Np):
        xmin = 0 if k==0 else pos_vec[k-1] + 1 # half-open interval (xmin included, xmax not included)
        xmax = Ns - Np + k + 1
        Ksites = list(range(0, xmin))
        Ksites_add = Ksites.copy()
        for ii, i in enumerate(range(xmin, xmax)):
            t0=time()
            # reference state        
            Ksites_add += [i]
            occ_vec_add = occ_vec[0:xmin] + [0]*ii + [1]
            Gnum = G[np.ix_(Ksites_add, Ksites_add)] - np.diag(occ_vec_add)
            Gdenom = G[np.ix_(Ksites, Ksites)] - np.diag(occ_vec[0:len(Ksites)])

            det_Gnum = np.linalg.det(Gnum)
            det_Gdenom = np.linalg.det(Gdenom)

            det_Gdenom_array[k, i] = det_Gdenom

            # Internal state used during low-rank update of conditional probabilities 
            # of the connnecting states. 

            # In case a cond. prob. of the reference state is zero:
            try:
                Gnum_inv = np.linalg.inv(Gnum) 
            except np.linalg.LinAlgError as e:
                print("ERROR: det_Gnum=%16.12f\n" % (det_Gnum), e)

            Gdenom_inv = np.linalg.inv(Gdenom)

            cond_prob_ref[k, i] = (-1) * det_Gnum / det_Gdenom
            t1 = time() 
            elapsed_ref += (t1 - t0)

            REF_SKIP = False 
            if np.isclose(cond_prob_ref[k,i], 0.0, atol=1e-8 / float(Ns)):
                counter_refskip += 1
                REF_SKIP = True 

            if (i,k) in S_connecting_states:
                det_Gnum_reuse.update({k : det_Gnum})

            # Now calculate the conditional probabilities for all states related 
            # to the reference state by one hop, using a low-rank update of `Gnum`
            # and `Gdenom`.
            t0_conn = time()
            for state_nr, (k_copy_, (r,s)) in enumerate(one_hop_info):
                if k_copy_ >= k:
                    # copy conditional probabilities rather than calculating them 
                    # Exit the loop; it is assumed that one-hop states are ordered according to increasing values 
                    # of k_copy_, i.e. the condition is also fulfilled for all subsequent states.
                    break 
                else: # k_copy < k  
                    # SOME SPECIAL CASES 
                    xmin_onehop = xs_pos[state_nr, k-1] + 1; xmax_onehop = xmax
                    if xmin_onehop == xmax_onehop-1 and r < i and s < i: 
                    # Every subsequent empty site needs to be occupied to accomodate all particles both in the reference state 
                    # and in the one-hop state. 
                        cond_prob_onehop[state_nr, k, i] = 1.0
                        cumul_sum_cond_prob_onehop[state_nr, k] = 1.0
                        continue

                    if r < s:
                        if i > s:
                            if not np.isclose(det_Gnum, 0.0, atol=1e-16): # don't invert a singular matrix 
                                counter_nonsingular += 1
                                cond = np.linalg.cond(Gdenom)
                                if cond > Gdenom_cond_max:
                                    Gdenom_cond_max = cond 

                                if k==(k_copy_+1):                                                   
                                    Gdenom_inv_, corr1 = adapt_Gdenom_inv(Gdenom_inv, Gglobal=G, r=r, s=s)
                                    corr2 = corr_factor_remove_r(Gdenom_inv_, r=r)
                                    corr_factor_Gdenom = corr1 * corr2
                                    corr_factor_Gnum = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s)
                                    corr_factor = corr_factor_Gnum / corr_factor_Gdenom                                                                      
                                else:
                                    corr_factor = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s) \
                                                    / corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s)
                                    print("corr_factor=", corr_factor)
                                if corr_factor < 0 and -corr_factor < 1e-12: corr_factor = 0.0
                                assert corr_factor >= 0, "state_nr=%d, k=%d, i=%i, corr_factor=%16.15f" % (state_nr, k, i, corr_factor)    
                                cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i] 

                            else: 
                                # As the numerator is singular, the conditional probabilities of the connecting states 
                                # should be calculated based on the matrix in the denominator, the inverse and determinant 
                                # of which are assumed to be known. The matrix in the denominator cannot be singular. 
                                t0 = time()
                                # # First check whether the conditional probabilities are already saturated.
                                # if np.isclose(sum(cond_prob_onehop[state_nr, k, xmin:i-1]), 1.0):  # CHECK: Why i-1 ? 
                                #     cond_prob_onehop[state_nr, k, i-1:] = 0.0
                                #     break

                                if cumul_sum_cond_prob_onehop[state_nr, k] > eps_norm_probs:
                                    cond_prob_onehop[state_nr, k, i-1:] = 0.0
                                    counter_skip += (xmax - i) 
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
                                    det_Gnum_ = corr4 * corr3 * corr1 * det_Gdenom
                                    #assert np.isclose(det_Gnum_, np.linalg.det(Gnum_))
                                    #cond_prob_onehop[state_nr, k, i] = (-1) * np.linalg.det(Gnum_) / np.linalg.det(Gdenom_)   
                                    cond_prob_onehop[state_nr, k, i] = (-1) * det_Gnum_ / (corr_factor_Gdenom * det_Gdenom) 
                                    #print("cond_prob_onehop[state_nr, k, i] = ", cond_prob_onehop[state_nr, k, i], np.linalg.det(Gnum_))

                                else:
                                    # connecting state and reference state have the same support in the denominator 
                                    ### Gdenom_inv = np.linalg.inv(Gdenom) OK
                                    corr_factor_Gdenom = corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s)
                                    #Gnum_ = adapt_singular_Gnum(Gnum, r=r, s=s, i=i) # not useful  
                                    det_Gnum_ = det_Gnum_from_Gdenom(Gdenom_inv, det_Gdenom, Gglobal=G, r=r, s=s, xmin=xmin, i=i)
                                    #assert np.isclose(det_Gnum_, np.linalg.det(Gnum_))
                                    cond_prob_onehop[state_nr, k, i] = (-1) * det_Gnum_ / (det_Gdenom * corr_factor_Gdenom)
                                    #cond_prob_onehop[state_nr, k, i] = (-1) * np.linalg.det(Gnum_) / (det_Gdenom * corr_factor_Gdenom)

                                counter_singular += 1
                                #print("SSingular, detGnum=", det_Gnum, "detGdenom=", det_Gdenom, "cond_prob_ref[k, i]=", cond_prob_ref[k, i])
                                #print("det_Gnum_=", det_Gnum_, "corr_factor_Gdenom=", corr_factor_Gdenom)

                                t1 = time()
                                elapsed_singular += (t1 - t0)
                    elif r > s: 
                            # The support is larger than in the reference state. One needs to calculate (r-s)
                            # more conditional probabilities than in the reference state. 
                            # In other words, here,  i not in (xmin, xmax). 
                            # The case i == r is special. 

                            if not np.isclose(det_Gnum, 0.0, atol=1e-16): # don't invert a singular matrix                                                  
                                counter_nonsingular += 1

                                cond = np.linalg.cond(Gdenom)
                                if cond > Gdenom_cond_max:
                                    Gdenom_cond_max = cond 

                                if k==(k_copy_+1):
                                    det_Gdenom_ = det_Gnum_reuse.get(k-1)
                                    if i==(r+1):
                                        # Actually we wish to calculate i==r, but since such an i is not in (xmin, xmax),
                                        # it will never appear in the iteration. Therefore this case is treated explicitly 
                                        # here. Since this case does not appear for the reference state, a "correction factor"
                                        # is not calculated, instead the cond. prob. is calculated directly:
                                        det_Gnum_ = det_Gdenom * corr_factor_add_s(Gdenom_inv, s=s)
                                        cond_prob_onehop[state_nr, k, i-1] = (-1) * det_Gnum_ / det_Gdenom_
                                        cumul_sum_cond_prob_onehop[state_nr, k] += cond_prob_onehop[state_nr, k, i-1]
                                    if i > r:  
                                        corr_factor = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s) \
                                                * (det_Gdenom / det_Gdenom_)
                                        cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]
    
                                else:
                                    corr_factor = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s) \
                                                    / corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s)     
                                    if corr_factor < 0 and -corr_factor < 1e-8: corr_factor = 0.0
                                    assert corr_factor >= 0, "state_nr=%d, k=%d, i=%i, corr_factor=%16.15f" % (state_nr, k, i, corr_factor)    
                                    cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]     

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
                                if cumul_sum_cond_prob_onehop[state_nr, k] > eps_norm_probs:
                                    cond_prob_onehop[state_nr, k, i-1:] = 0.0
                                    counter_skip += (xmax - i)
                                    continue                                    
                                counter_singular += 1
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

                                        cumul_sum_cond_prob_onehop[state_nr, k] += cond_prob_onehop[state_nr, k, i-1]
                                        #cond_prob_onehop[state_nr, k, i-1] = (-1) * np.linalg.det(Gnum_) / det_Gdenom_
                                    if i > r:                                     
                                        #Gnum_ = adapt_singular_Gnum2(Gnum, r=r, s=s, i=i)
                                        #print("Gnum_.shape=", Gnum_.shape)
                                        det_Gnum_ = det_Gnum_from_Gdenom(Gdenom_inv, det_Gdenom, Gglobal=G, r=r, s=s, xmin=xmin, i=i)
                                        ##print("det_Gnum_=", det_Gnum_, "np.linalg.det(Gnum_)=", np.linalg.det(Gnum_))
                                        #assert np.isclose(det_Gnum_, np.linalg.det(Gnum_) )
                                        cond_prob_onehop[state_nr, k, i] = (-1) * det_Gnum_ / det_Gdenom_

                                        #cond_prob_onehop[state_nr, k, i] = (-1) * np.linalg.det(Gnum_) / det_Gdenom_  
                                else:
                                    #print("singular numerator, k != k_copy+1, k=", k, "i=", i)
                                    corr_factor_Gdenom = corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s) # OK
                                    #Gnum_ = adapt_singular_Gnum2(Gnum, r=r, s=s, i=i) 
                                    det_Gnum_ = det_Gnum_from_Gdenom(Gdenom_inv, det_Gdenom, Gglobal=G, r=r, s=s, xmin=xmin, i=i)
                                    #print("det_Gnum_=", det_Gnum_, "np.linalg.det(Gnum_)=", np.linalg.det(Gnum_))  
                                    cond_prob_onehop[state_nr, k, i] = (-1) * det_Gnum_ / ( det_Gdenom * corr_factor_Gdenom)           
                                    #cond_prob_onehop[state_nr, k, i] = (-1) * np.linalg.det(Gnum_) / ( det_Gdenom * corr_factor_Gdenom)
                                    
                                t1 = time()
                                elapsed_singular += (t1 - t0)

                    if state_nr == 0 and k == 4:
                        print("cond_prob_onehop[state_nr, k, i]=", cond_prob_onehop[state_nr, k, i], r, s, i, det_Gnum)

                    cumul_sum_cond_prob_onehop[state_nr, k] += cond_prob_onehop[state_nr, k, i]

            t1_conn = time()
            elapsed_connecting_states += (t1_conn - t0_conn)                    

    for state_nr, (k_copy_, (r,s)) in enumerate(one_hop_info):
        for k in range(Np):
            if k > k_copy_:
                print("k=", k, "state_nr=", state_nr, "cumul=", cumul_sum_cond_prob_onehop[state_nr,k])
                #print("ref_conf=", ref_conf)
                print("1hop sta=", xs[state_nr])
                assert np.isclose(cumul_sum_cond_prob_onehop[state_nr,k], 1.0, atol=ATOL), "cumul_sum_cond_prob_onehop[state_nr=%d, k=%d]=%16.10f" % (state_nr, k, cumul_sum_cond_prob_onehop[state_nr,k])


    fh = open("cond_prob_ref.dat", "w")
    fh2 = open("det_Gdenom_array.dat", "w")
    for k in range(cond_prob_ref.shape[0]):
        for i in range(cond_prob_ref.shape[1]):
            fh.write("%d %d %e\n" % (k, i, cond_prob_ref[k, i]))
            fh2.write("%d %d %e\n" % (k, i, det_Gdenom_array[k, i]))
    fh.close()
    fh2.close()


    # Check 
    copy_cond_probs(cond_prob_ref, cond_prob_onehop, one_hop_info)

    for state_nr in range(cond_prob_onehop.shape[0]):
        fh = open("cond_prob_onehop%d.dat" % (state_nr), "w")
        fh.write("# ref_state ["+" ".join(str(item) for item in ref_conf)+"]\n")
        fh.write("# 1hop      ["+" ".join(str(item) for item in xs[state_nr])+"]\n")
        for k in range(cond_prob_onehop.shape[1]):
            for i in range(cond_prob_onehop.shape[2]):
                fh.write("%d %d %20.19f\n" % (k, i, cond_prob_onehop[state_nr, k, i]))
        fh.close()    

    log_probs = cond_prob2log_prob(xs, cond_prob_onehop)
    log_prob_ref = cond_prob2log_prob([ref_conf], cond_prob_ref[None,:])

    print("log_prob_ref=", log_prob_ref)
    print("ref_conf=", ref_conf)
    for i in range(num_connecting_states):
        print("state_nr=", i)
        print("xs=      ", xs[i])
        print(log_probs[i], SDsampler.log_prob([xs[i]]).item())
        print(np.exp(log_probs[i] - log_prob_ref))
        assert np.isclose( np.exp(log_probs[i] - log_prob_ref), np.exp(SDsampler.log_prob([xs[i]]).item()  - log_prob_ref))


    print("elapsed_ref=", elapsed_ref)
    print("elapsed_connecting_states=", elapsed_connecting_states)
    print("elapsed_singular=", elapsed_singular)
    print("counter_singular=", counter_singular)
    print("counter_skip=", counter_skip)
    print("counter_refskip=", counter_refskip)
    print("counter_nonsingular=", counter_nonsingular)
    print("elapsed_adapt=", elapsed_adapt)

def _test():
    import doctest
    doctest.testmod(verbose=False)

if __name__ == "__main__":
    _test()
