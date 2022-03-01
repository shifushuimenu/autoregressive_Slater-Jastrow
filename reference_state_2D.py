# TODO:
#     - corr3_Gnum_from_Gdenom(): What if the matrix S is singular ? linalg.inv(S) raises an error 
#     - The case of a singular numerator matrix occurs very often (more than 50 %). 
#     - The matrix CC in lowrank_update_inv_addremove_rs() sometimes has extremely high condition numbers. 

import numpy as np
from time import time 

from bitcoding import *
from Slater_Jastrow_simple import kinetic_term, Lattice1d, Lattice_rectangular

from lowrank_update import * 
from k_copy import * 

from test_suite import *

# Calculate the conditional probabilities of the reference state
Ns = 20; Np = 10    # Ns=50; Np=19 -> normalization only up to 10e-5. Are there numerical instabilities ? 
_, U = prepare_test_system_zeroT(Nsites=Ns, potential='none', Nparticles=Np)
P = U[:, 0:Np]
G = np.eye(Ns) - np.matmul(P, P.transpose(-1,-2))

eps_norm_probs = 1.0 - 1e-7 # Note: np.isclose(1.0 - 1e-5, 1.0) == True

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
    
ref_conf = gen_random_config(Ns, Np)
ref_I = bin2int(ref_conf).numpy() # ATTENTION: Wrong results for too large bitarrays !
#ref_I = bin2int(np.array([1,0,1,0,1,0,0,1,1]))

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
# # END: JUST FOR TESTING THE SCALING 


# special case of 1d n.n. hopping matrix 
assert np.all([abs(r-s) == 1 or abs(r-s) == Ns-1 for r,s in rs_pos])
k_copy = calc_k_copy(rs_pos, ref_conf)
one_hop_info = list(zip(k_copy, rs_pos))

s_pos = list(s for (r,s) in rs_pos if s < r)
# S_connecting_states = ((s0, k0), (s1, k1), ...)
#     ki-th particle sits at position si. 
S_connecting_states = list(zip(s_pos, [kk for idx, kk in enumerate(k_copy) if rs_pos[idx][1] < rs_pos[idx][0]]))
det_Gnum_reuse = dict()

cond_prob_ref = np.zeros((Np, Ns))
cond_prob_onehop = np.zeros((num_connecting_states, Np, Ns))
cumul_sum_cond_prob_onehop = np.zeros((num_connecting_states, Np))

Ksites = []
occ_vec = list(ref_conf)
assert type(occ_vec) == type(list()) # use a list, otherwise `occ_vec[0:xmin] + [1]` will result in `[]`. 
pos_vec = bin2pos(ref_conf)


for k in range(Np):
    xmin = 0 if k==0 else pos_vec[k-1] + 1 # half-open interval (xmin included, xmax not included)
    xmax = Ns - Np + k + 1
    Ksites = list(range(0, xmin))
    Ksites_add = Ksites.copy()
    for ii, i in enumerate(range(xmin, xmax)):
        # reference state        
        Ksites_add += [i]
        occ_vec_add = occ_vec[0:xmin] + [0]*ii + [1]
        Gnum = G[np.ix_(Ksites_add, Ksites_add)] - np.diag(occ_vec_add)
        Gdenom = G[np.ix_(Ksites, Ksites)] - np.diag(occ_vec[0:len(Ksites)])

        det_Gnum = np.linalg.det(Gnum)
        det_Gdenom = np.linalg.det(Gdenom)

        # Internal state used during low-rank update of conditional probabilities 
        # of the connnecting states. 
        Gnum_inv = np.linalg.inv(Gnum)   # OK 
        Gdenom_inv = np.linalg.inv(Gdenom) # OK

        cond_prob_ref[k, i] = (-1) * det_Gnum / det_Gdenom

        REF_SKIP = False 
        if np.isclose(cond_prob_ref[k,i], 0.0, atol=1e-8 / float(Ns)):
            REF_SKIP = True 

        if (i,k) in S_connecting_states:
            det_Gnum_reuse.update({k : det_Gnum})

        # Now calculate the conditional probabilities for all states related 
        # to the reference state by one hop, using a low-rank update of `Gnum`
        # and `Gdenom`.
        for state_nr, (k_copy_, (r,s)) in enumerate(one_hop_info):
            if k_copy_ >= k:
                break # copy conditional probabilities rather than calculating them 
            else:
                if r > s:
                    # For i \in [s+1, r] calculations are based on the reference state at k-1.
                    # The positions i \in [s+1, i_k] have no correspondence in the reference state. 
                    # (i_k is the position of the k-th particle.) They need to be calculated additionally. 
                    # They are calculated together with i==xmin.
                    if i == xmin:
                        for i2 in np.arange(s+1, i_k+1):
                            det_Gnum_reuse.get(k-1)
                            cond_prob_onehop[state_nr, k, i2] = 0.0

                elif s > r: 
                    pass
                                
            
            cumul_sum_cond_prob_onehop[state_nr, k] += cond_prob_onehop[state_nr, k, i]


for state_nr, (k_copy_, (r,s)) in enumerate(one_hop_info):
    for k in range(Np):
        if k > k_copy_:
            assert np.isclose(cumul_sum_cond_prob_onehop[state_nr,k], 1.0)
assert np.isclose(np.sum(cond_prob_ref, axis=1), np.ones((Np,1))).all()
for state_nr in range(num_connecting_states):
    for k in range(Np):
        if k > k_copy[state_nr]:
            print("sum(cond_prob_onehop=", np.sum(cond_prob_onehop[state_nr, k, :]))
            assert np.isclose(np.sum(cond_prob_onehop[state_nr, k, :]), 1.0)

fh = open("cond_prob_ref.dat", "w")
for k in range(cond_prob_ref.shape[0]):
    for i in range(cond_prob_ref.shape[1]):
        fh.write("%d %d %e\n" % (k, i, cond_prob_ref[k, i]))
fh.close()


for state_nr in range(cond_prob_onehop.shape[0]):
    fh = open("cond_prob_onehop%d.dat" % (state_nr), "w")
    for k in range(cond_prob_onehop.shape[1]):
        for i in range(cond_prob_onehop.shape[2]):
            fh.write("%d %d %16.15f\n" % (k, i, cond_prob_onehop[state_nr, k, i]))
    fh.close()
