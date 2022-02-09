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
k_copy = (-1, # don't copy any conditional probabilities  
           0,
           0,
           1,
           1,
           2)

# The "one-hop" config is generated from the reference state by 
# removing a particle at position `r` and putting it 
# at position `s`. 
rs_pos = ( (0,1), 
           (2,1),
           (2,3),
           (4,3),
           (4,5),
           (7,6) )

one_hop_info = dict(k_copy=k_copy, rs_pos=rs_pos, xs=one_hop_conf)

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


# Calculate the conditional probabilities of the reference state
Ns = 9; Np = 5
_, U = prepare_test_system_zeroT(Nsites=Ns, potential='none', Nparticles=Np)
P = U[:, 0:Np]
G = np.eye(Ns) - np.matmul(P, P.transpose(-1,-2))

print("G_global=", G[0:4, 0:4])

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
        r=0; s=1
        if k > 0 and i > s: 
            if not np.isclose(np.linalg.det(Gnum), 0.0):

                print(k, ii, i, cond_prob_ref[k,i])
                print("Gnum.shape=", Gnum.shape)
                print("Gdenom.shape=", Gdenom.shape)
                Gnum_inv = np.linalg.inv(Gnum)
                if k==1:                     
                    corr_factor = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s) \
                                  * np.linalg.det(Gdenom)
                    Gdenom_ = adapt_Gdenom(Gnum, r=r, s=s)
                    corr_factor /= np.linalg.det(Gdenom_)
                else:
                    Gdenom_inv = np.linalg.inv(Gdenom)
                    corr_factor = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s) \
                                    / corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s) 
                cond_prob_onehop[0, k, i] = corr_factor * cond_prob_ref[k, i]
            else: 
                cond_prob_onehop[0, k, i] = 0.0

assert np.isclose(np.sum(cond_prob_ref, axis=1), np.ones((Np,1))).all()
print("cond_prob_onehop[0, :, :]=", cond_prob_onehop[0, :, :])