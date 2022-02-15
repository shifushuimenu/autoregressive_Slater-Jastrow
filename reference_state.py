from curses import KEY_SCOPY
import numpy as np
from test_suite import prepare_test_system_zeroT
from bitcoding import *
from Slater_Jastrow_simple import kinetic_term, Lattice1d
from block_update_numpy import ( block_update_inverse,
                           block_update_det_correction )

from time import time 

np.random.seed(422)


def exclude_invalid_connecting_states(hop_from_to, states_I, matrix_elem):
    STATE_EXISTS = np.where(matrix_elem != 0)
    for _ in range(matrix_elem.shape[1] - len(STATE_EXISTS[0])):
        hop_from_to.remove((-1,-1))
    matrix_elem = matrix_elem[STATE_EXISTS]
    states_I = states_I[STATE_EXISTS]

    return (hop_from_to, states_I, matrix_elem)

#alias 
valid_states = exclude_invalid_connecting_states 

# Calculate k_copy
def calc_k_copy(hop_from_to, ref_state_I, ns):
    """
        ONLY VALID FOR 1D N.N. HOPPING MATRIX.

        `k_copy` indicates the component up to which (inclusive)
        the conditional probabilities are identical to those 
        of the reference state (such that they can be copied). The index into 
        k_copy is the reference state number. 
        For example: 
            k_copy = (  0, # The first component is conditionally independent, it can always be copied from the reference state. 
                        1, # copy conditional probs up to (inclusive) the second component 
                        1,
                        2, # copy conditional probs up to (inclusive) the third component
                        2, 
                        3 )

        Example:
        >>> from Slater_Jastrow_simple import Lattice1d
        >>> Ns=9; l1d = Lattice1d(ns=Ns); I = [2**8 + 2**7 + 2**4 + 2**2 + 2**0]
        >>> hop_from_to, states_I, matrix_elem = valid_states(*kinetic_term(I, l1d))
        >>> k_copy = (0, 1, 1, 2, 2, 3)
        >>> k_copy == calc_k_copy(hop_from_to, I, Ns)
        True
    """
    pos_ref_state = int2pos(ref_state_I, ns=ns)
    num_connecting_states = len(hop_from_to)
    k_copy = np.zeros((num_connecting_states,), dtype=int)

    for i in range(num_connecting_states):
        (r,s) = hop_from_to[i]
        ii = 0
        for pos in pos_ref_state.flatten()[ii:]:
            # Note: it is assumed that `hop_from_to` is already ordered in increasing order of `r`. 
            if r == pos:
                k_copy[i] = ii
                break 
            ii = ii + 1

    assert monotonically_increasing(k_copy)

    return tuple(k_copy)


def monotonically_increasing(y):
    r = True 
    for i in range(len(y)-1):
        r = r and (y[i] <= y[i+1])
        if not r: return r 
    return r 









def corr_factor_removeadd_rs(Ainv, r, s):
    """ 
        Correction factor to 
             det(G_{K,K} - N_K)
        with N_K = (n_0, n_1, ..., n_{K-1}) where n_r = 1 and n_s = 0
        and a particle is moved such that after the update n_r = 0 and n_s = 1.         
    """
    return (1 + Ainv[r,r]) * (1 - Ainv[s,s]) + Ainv[r,s]*Ainv[s,r]

def corr_factor_add_s(Ainv, s):
    """add a particle at position s without removing any particle """
    return (1 - Ainv[s,s])

def corr_factor_remove_r(Ainv, r):
    """remove a particle at position r without adding any particle"""
    return (1 + Ainv[r,r])

def adapt_Gdenom(Gnum, r, s):
    assert s > r 
    G = Gnum[np.ix_(list(range(0, s+1)), list(range(0, s+1)))]
    G[r,r] = G[r,r] + 1
    G[s,s] = G[s,s] - 1
    return G

def adapt_Gdenom_inv(Gdenom_inv, Gglobal, r, s):
    assert s > r 
    assert Gdenom_inv.shape == (s, s)
    assert s == r +1 
    # put an additional particle at position s 
    Gdenom_inv_ = block_update_inverse(Ainv=Gdenom_inv, B=Gglobal[0:s, s][:, None], C=Gglobal[s,0:s][None, :], D=Gglobal[s,s][None, None] - 1)
    corr = block_update_det_correction(Ainv=Gdenom_inv, B=Gglobal[0:s, s][:, None], C=Gglobal[s,0:s][None, :], D=Gglobal[s,s][None, None] - 1)
    return Gdenom_inv_, corr 
    pass

def reduce_Gdenom(Gdenom, r, s):
    assert r > s 
    assert Gdenom.shape == (r+1, r+1)
    G = Gdenom[np.ix_(list(range(0, s+1)), list(range(0, s+1)))]
    G[s,s] = G[s,s] - 1
    return G

def reduce_Gnum(Gdenom, r, s): # yes, Gdenom is the argument !
    assert r > s 
    assert Gdenom.shape == (r+1, r+1)
    G = Gdenom[np.ix_(list(range(0, r+1)), list(range(0, r+1)))]
    G[s,s] = G[s,s] - 1
    # G[r,r] = G[r,r] # Now, there is a particle both at position r and s. 
    return G

# The following three functions are for the case of a singular numerator.
def Gdenom_from_Gdenom(Gdenom, r, s):
    assert r > s 
    assert Gdenom.shape == (r+1, r+1)
    G = Gdenom[np.ix_(list(range(0, r)), list(range(0, r)))]
    G[s,s] = G[s,s] - 1
    return G

def Gnum_from_Gdenom_ieqrp1(Gdenom, r, s, i):
    assert r > s 
    assert Gdenom.shape == (r+1, r+1)
    assert i==(r+1)
    G = Gdenom[np.ix_(list(range(0, r+1)), list(range(0, r+1)))]        
    G[s,s] = G[s,s] - 1
    return G


def adapt_singular_Gnum(Gnum, r, s, i):
    # THIS IS NOT USEFUL
    assert s > r 
    assert i > r
    G = Gnum.copy()
    G[r,r] = G[r,r] + 1
    G[s,s] = G[s,s] - 1
    return G

def adapt_singular_Gnum2(Gnum, r, s, i):
    # THIS IS NOT USEFUL
    assert r > s
    assert i > r
    G = Gnum.copy()
    G[r,r] = G[r,r] + 1 
    G[s,s] = G[s,s] - 1
    return G


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
Ns = 60; Np = 30   # Ns=12; Np=2 -> problem with corr_factor_removeadd_rs() !!!!!!!!
_, U = prepare_test_system_zeroT(Nsites=Ns, potential='none', Nparticles=Np)
P = U[:, 0:Np]
G = np.eye(Ns) - np.matmul(P, P.transpose(-1,-2))

def gen_random_config_I(Ns, Np):
    """generate a random reference state of fixed particle number"""
    config = np.zeros((Ns,), dtype=int) 
    config[0] = 1; config[-1] = 1 # !!!!! REMOVE: Make sure no hopping across p.b.c can occur. 
    counter = 0
    while counter < Np-2:
        pos = np.random.randint(low=0, high=Ns, size=1)
        if config[pos] != 1:
            config[pos] = 1
            counter += 1 
    return bin2int(config).numpy()

ref_I = gen_random_config_I(Ns, Np)
#ref_I = bin2int(np.array([1,0,1,0,1,0,0,1,1]))
ref_conf = int2bin(ref_I, ns=Ns)

# # The "one-hop" config is generated from the reference state by 
# # removing a particle at position `r` and putting it 
# # at position `s`. 
# e.g.: rs_pos = ( (0,1), 
#                  (2,1),
#                  (2,3),
#                  (4,3),
#                  (4,5),
#                  (7,6) )

l1d = Lattice1d(ns=Ns)
# `states_I` are only the connecting states, the reference state is not included 
rs_pos, states_I, _ = valid_states(*kinetic_term([ref_I], l1d))
num_connecting_states = len(states_I)
xs = int2bin(states_I, ns=Ns)
# special case of 1d n.n. hopping matrix 
assert np.all([abs(r-s) == 1 or abs(r-s) == Ns-1 for r,s in rs_pos])
print("reference state=")
print(int2bin(ref_I, ns=Ns))
print("connecting_states=")
print(xs)
print("rs_pos=", rs_pos)
k_copy = calc_k_copy(rs_pos, ref_I, ns=Ns)
print("k_copy=", k_copy)
one_hop_info = list(zip(k_copy, rs_pos))

s_pos = list(s for (r,s) in rs_pos if s < r)
# S_connecting_states = ((s0, k0), (s1, k1), ...)
#     ki-th particle sits at position si. 
S_connecting_states = list(zip(s_pos, [kk for idx, kk in enumerate(k_copy) if rs_pos[idx][1] < rs_pos[idx][0]]))
det_Gnum_reuse = dict()

cond_prob_ref = np.zeros((Np, Ns))
cond_prob_onehop = np.zeros((num_connecting_states, Np, Ns))

Ksites = []
occ_vec = list(ref_conf)
assert type(occ_vec) == type(list()) # use a list, otherwise `occ_vec[0:xmin] + [1]` will result in `[]`. 
pos_vec = bin2pos(ref_conf)

elapsed_ref = 0.0
elapsed_connecting_states = 0.0
elapsed_singular = 0.0
elapsed_adapt = 0.0

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
        
        # Internal state used during low-rank update of conditional probabilities 
        # of the connnecting states. 
        Gnum_inv = np.linalg.inv(Gnum)   # OK 
        Gdenom_inv = np.linalg.inv(Gdenom) # OK

        det_Gnum = np.linalg.det(Gnum)
        det_Gdenom = np.linalg.det(Gdenom)
        cond_prob_ref[k, i] = (-1) * det_Gnum / det_Gdenom
        t1 = time() 
        elapsed_ref += (t1 - t0)
        if (i,k) in S_connecting_states:
            det_Gnum_reuse.update({k : det_Gnum})

        # Now calculate the conditional probabilities for all states related 
        # to the reference state by one hop, using a low-rank update of `Gnum`
        # and `Gdenom`.
        t0_conn = time()
        for state_nr, (k_copy_, (r,s)) in enumerate(one_hop_info):
            if k_copy_ >= k:
                break # copy conditional probabilities rather than calculating them 
            else: # k_copy < k
                if s > r:
                    if i > s:
                        if not np.isclose(det_Gnum, 0.0): # don't invert a singular matrix 
                            if k==(k_copy_+1):               
                                # everythink OK      
                                corr_factor_Gnum = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s)
                                Gdenom_inv_, corr1 = adapt_Gdenom_inv(Gdenom_inv, Gglobal=G, r=r, s=s)
                                corr2 = corr_factor_remove_r(Gdenom_inv_, r=r)
                                corr_factor_Gdenom = corr1 * corr2
                                corr_factor = corr_factor_Gnum / corr_factor_Gdenom
                            else:
                                corr_factor = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s) \
                                                / corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s) # OK
                            cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]
                        else: 
                            # As the numerator is singular, the conditional probabilities of the connecting states 
                            # should be calculated based on the matrix in the denominator, the inverse and determinant 
                            # of which are assumed to be known. The matrix in the denominator cannot be singular. 

                            # First check whether the conditional probabilities are already saturated.
                            if np.isclose(sum(cond_prob_onehop[state_nr, k, 0:i-1]), 1.0):  # CHECK: Why i-1 ? 
                                cond_prob_onehop[state_nr, k, i-1:] = 0.0
                                break
                            # print("numerator 1 singular, k=", k, " i=", i, "sum(cond_prob_ref)=", sum(cond_prob_ref[k, 0:i+1]), sum(cond_prob_onehop[state_nr, k, 0:i]))
                            t0 = time()
                            if k==(k_copy_+1):
                                Gdenom_ = adapt_Gdenom(Gnum, r=r, s=s)
                                Gnum_ = Gnum_from_Gdenom3(Gdenom_, Gglobal=G, r=r, s=s, i=i)
                                cond_prob_onehop[state_nr, k, i] = (-1) * np.linalg.det(Gnum_) / np.linalg.det(Gdenom_)                            
                                #print("cond_prob_onehop[state_nr, k, i] = ", cond_prob_onehop[state_nr, k, i], np.linalg.det(Gnum_))
                            else:
                                # connecting state and reference state have the same support in the denominator 
                                ### Gdenom_inv = np.linalg.inv(Gdenom) OK
                                corr_factor_Gdenom = corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s)
                                Gnum_ = adapt_singular_Gnum(Gnum, r=r, s=s, i=i) # not useful                                
                                cond_prob_onehop[state_nr, k, i] = (-1) * np.linalg.det(Gnum_) / (det_Gdenom * corr_factor_Gdenom)
                            t1 = time()
                            elapsed_singular += (t1 - t0)
                elif r > s: 
                        # The support is larger than in the reference state. One needs to calculate (r-s)
                        # more conditional probabilities than in the reference state. 
                        # In other words, here,  i not in (xmin, xmax). 
                        # The case i == r is special. 
                        if not np.isclose(det_Gnum, 0.0): # don't invert a singular matrix                                                  
                            if k==(k_copy_+1):
                                det_Gdenom_ = det_Gnum_reuse.get(k-1)
                                if i==(r+1):
                                    # Actually we wish to calculate i==r, but since such an i is not in (xmin, xmax),
                                    # it will never appear in the iteration. Therefore this case is treated explicitly 
                                    # here. Since this case does not appear for the reference state, a "correction factor"
                                    # is not calculated, instead the cond. prob. is calculated directly:
                                    det_Gnum_ = det_Gdenom * corr_factor_add_s(Gdenom_inv, s=s) # OK
                                    cond_prob_onehop[state_nr, k, i-1] = (-1) * det_Gnum_ / det_Gdenom_ # OK
                                if i > r:  
                                    corr_factor = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s) \
                                            * (det_Gdenom / det_Gdenom_)   # OK                          
                            else:
                                corr_factor = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s) \
                                                / corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s) # OK
                            cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]
                        else:
                            # As the numerator is singular, the conditional probabilities of the connecting states 
                            # should be calculated based on the matrix in the denominator, the inverse and determinant 
                            # of which are assumed to be known. The matrix in the denominator cannot be singular. 
                            # First check whether the conditional probabilities are already saturated.
                            if np.isclose(sum(cond_prob_onehop[state_nr, k, 0:i-1]), 1.0): # CHECK: Why i-1 ? 
                                cond_prob_onehop[state_nr, k, i-1:] = 0.0
                                break                          
                            # print("numerator 2 singular, k=", k, " i=", i, "sum(cond_prob)=", sum(cond_prob_ref[k, 0:i+1]))
                            t0 = time()
                            if k==(k_copy_+1):
                                Gdenom_ = Gdenom_from_Gdenom(Gdenom, r=r, s=s)
                                if i==(r+1):
                                    Gnum_ = Gnum_from_Gdenom_ieqrp1(Gdenom, r=r, s=s, i=i)           
                                    cond_prob_onehop[state_nr, k, i-1] = (-1) * np.linalg.det(Gnum_) / np.linalg.det(Gdenom_)                                                             
                                if i > r: 
                                    Gnum_ = adapt_singular_Gnum2(Gnum, r=r, s=s, i=i)
                                    cond_prob_onehop[state_nr, k, i] = (-1) * np.linalg.det(Gnum_) / np.linalg.det(Gdenom_)                  
                            else:
                                corr_factor_Gdenom = corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s) # OK
                                Gnum_ = adapt_singular_Gnum2(Gnum, r=r, s=s, i=i) 
                                cond_prob_onehop[state_nr, k, i] = (-1) * np.linalg.det(Gnum_) / ( det_Gdenom * corr_factor_Gdenom)
                            t1 = time()
                            elapsed_singular += (t1 - t0)
        t1_conn = time()
        elapsed_connecting_states += (t1_conn - t0_conn)                    

assert np.isclose(np.sum(cond_prob_ref, axis=1), np.ones((Np,1))).all()
print("sum(cond_prob_ref=", np.sum(cond_prob_ref, axis=1))
for state_nr in range(num_connecting_states):
    for k in range(Np):
        if k > k_copy[state_nr]:
            print("state_nr=", state_nr, "k=", k)
            print("cond_prob_onehop[%d, %d, :]=<"%(state_nr, k), cond_prob_onehop[state_nr, k, :])
            print("sum(cond_prob_onehop=", np.sum(cond_prob_onehop[state_nr, k, :]))
            assert np.isclose(np.sum(cond_prob_onehop[state_nr, k, :]), 1.0)


print("elapsed_ref=", elapsed_ref)
print("elapsed_connecting_states=", elapsed_connecting_states)
print("elapsed_singular=", elapsed_singular)
print("elapsed_adapt=", elapsed_adapt)

def _test():
    import doctest
    doctest.testmod(verbose=False)

if __name__ == "__main__":
    _test()
