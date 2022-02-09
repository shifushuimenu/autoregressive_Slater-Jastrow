from curses import KEY_SCOPY
import numpy as np
from test_suite import prepare_test_system_zeroT

from bitcoding import *

xs = ([1,0,1,0,1,0,0,1,1],
      [0,1,1,0,1,0,0,1,1],
      [1,1,0,0,1,0,0,1,1],
      [1,0,0,1,1,0,0,1,1],
      [1,0,1,1,0,0,0,1,1],
      [1,0,1,0,0,1,0,1,1],
      [1,0,1,0,1,0,1,0,1])

# xs = filter(lambda x: torch.tensor(x), xs)
# xs = [torch.tensor(x) for x in xs]
ref_conf = xs[0]
one_hop_conf = xs[1:]

# `k_copy` indicates the component up to which (inclusive)
# the conditional probabilities are identical to those 
# of the reference state (such that they can be copied).
k_copy = ( 0, # The first component is conditionally independent, it can always be copied from the reference state. 
           1, # copy conditional probs up to (inclusive) the second component 
           1,
           2, # copy conditional probs up to (inclusive) the third component
           2, 
           3)

def monotonically_increasing(y):
    r = True 
    for i in range(len(y)-1):
        r = r and (y[i] <= y[i+1])
        if not r: return r 
    return r 

assert monotonically_increasing(k_copy)

# The "one-hop" config is generated from the reference state by 
# removing a particle at position `r` and putting it 
# at position `s`. 
rs_pos = ( (0,1), 
           (2,1),
           (2,3),
           (4,3),
           (4,5),
           (7,6) )

# special case of 1d n.n. hopping matrix 
assert np.all([abs(r-s) == 1 for r,s in rs_pos])

one_hop_info = list(zip(k_copy, rs_pos))

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

def reduce_Gdenom(Gdenom, r, s):
    assert r > s 
    assert Gdenom.shape == (r+1, r+1)
    G = Gdenom[np.ix_(list(range(0, s+1)), list(range(0, s+1)))]
    G[s,s] = G[s,s] - 1
    return G

def reduce_Gnum(Gdenom, r, s): # yes, Gdenom is the argument !
    assert r > s 
    assert Gdenom.shape == (r+1, r+1)
    G = Gnum[np.ix_(list(range(0, r+1)), list(range(0, r+1)))]
    G[s,s] = G[s,s] - 1
    # G[r,r] = G[r,r] # Now, there is a particle both at position r and s. 
    return G


# Calculate the conditional probabilities of the reference state
Ns = 9; Np = 5
_, U = prepare_test_system_zeroT(Nsites=Ns, potential='none', Nparticles=Np)
P = U[:, 0:Np]
G = np.eye(Ns) - np.matmul(P, P.transpose(-1,-2))

cond_prob_ref = np.zeros((Np, Ns))
cond_prob_onehop = np.zeros((len(xs[1:]), Np, Ns))

Ksites = []
occ_vec = ref_conf
pos_vec = bin2pos(ref_conf)
for k in range(Np):
    xmin = 0 if k==0 else pos_vec[k-1] + 1 # half-open interval (xmin included, xmax not included)
    xmax = Ns - Np + k + 1
    Ksites = list(range(0, pos_vec[k-1]+1))
    Ksites_add = Ksites.copy()
    for ii, i in enumerate(range(xmin, xmax)):
        Ksites_add += [i]
        occ_vec_add = occ_vec[0:pos_vec[k-1]+1] + [0]*ii + [1]
        Gnum = G[np.ix_(Ksites_add, Ksites_add)] - np.diag(occ_vec_add)
        Gdenom = G[np.ix_(Ksites, Ksites)] - np.diag(occ_vec[0:len(Ksites)])

        cond_prob_ref[k, i] = (-1) * np.linalg.det(Gnum) / np.linalg.det(Gdenom)

        # Now calculate the conditional probabilities for all states related 
        # to the reference state by one hop, using a low-rank update of `Gnum`
        # and `Gdenom`.
        for state_nr, (k_copy, (r,s)) in enumerate(one_hop_info):
            print("state_nr=", state_nr)
            if k_copy >= k:
                break # copy conditional probabilities rather than calculating them 
            else: # k_copy < k
                if s > r:
                    if i > s:
                        if not np.isclose(np.linalg.det(Gnum), 0.0): # don't invert a singular matrix 
                            print(k, ii, i, cond_prob_ref[k,i])
                            Gnum_inv = np.linalg.inv(Gnum)
                            if k==(k_copy+1):                     
                                corr_factor = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s) \
                                            * np.linalg.det(Gdenom)
                                Gdenom_ = adapt_Gdenom(Gnum, r=r, s=s)
                                corr_factor /= np.linalg.det(Gdenom_)
                            else:
                                Gdenom_inv = np.linalg.inv(Gdenom)
                                corr_factor = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s) \
                                                / corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s) 
                            cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]
                        else: 
                            cond_prob_onehop[state_nr, k, i] = 0.0
                elif r > s: 
                    # The support is larger than in the reference state. One needs to calculate (r-s)
                    # more conditional probabilities than in the reference state. 
                    # In other words, here,  i not in (xmin, xmax). 
                    # ... TODO ...
                    # The case i == r is special. 
                    if not np.isclose(np.linalg.det(Gnum), 0.0): # don't invert a singular matrix      
                        Gnum_inv = np.linalg.inv(Gnum)                                                   
                        if k==(k_copy+1):
                            Gdenom_ = reduce_Gdenom(Gdenom, r=r, s=s)
                            if i==(r+1):
                                # Actually we wish to calculate i==r, but since such an i is not in (xmin, xmax),
                                # it will never appear in the iteration. Therefore this case is treated explicitly 
                                # here. Since this case does not appear for the reference state, a "correction factor"
                                # is not calculated, instead the cond. prob. is calculated directly:
                                Gnum_ = reduce_Gnum(Gdenom, r=r, s=s)
                                cond_prob_onehop[state_nr, k, i-1] = (-1) * np.linalg.det(Gnum_) / np.linalg.det(Gdenom_)
                                pass
                            if i > r:  
                                corr_factor = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s) \
                                        * (np.linalg.det(Gdenom) / np.linalg.det(Gdenom_))
                        else:
                            Gdenom_inv = np.linalg.inv(Gdenom)
                            corr_factor = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s) \
                                            / corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s) 
                        cond_prob_onehop[state_nr, k, i] = corr_factor * cond_prob_ref[k, i]
                    else:
                        cond_prob_onehop[state_nr, k, i] = 0.0


assert np.isclose(np.sum(cond_prob_ref, axis=1), np.ones((Np,1))).all()
for state_nr in range(6):
    print("cond_prob_onehop[%d, :, :]="%(state_nr), cond_prob_onehop[state_nr, :, :])
    print(np.sum(cond_prob_onehop[state_nr, :, :], axis=1))