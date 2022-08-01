"""Exact diagonalization of the 1d t-V model (spin-polarized fermions)"""

import numpy as np
from bitcoding import *
from physics import ( kinetic_term2, 
    Lattice1d, Lattice_rectangular )
from scipy.special import binom 

import matplotlib.pyplot as plt 


t_hop = 1.0 # t_hop > 0, since kinetic_term() provides already a minus sign 
V_nnint = 0.0

for V_nnint in (0.0,): #np.linspace(3.0, 3.0, 1):
    Nx = 4
    Ny = 1
    Ns = Nx*Ny # 13
    Np = 2 #5


    dimH = int(binom(Ns, Np))
    ###print("dimH=", dimH)
    lattice = Lattice1d(ns=Ns)
    #lattice = Lattice_rectangular(Nx, Ny)

    # build the basis 
    basis_dict = {}
    invbasis_dict = {}
    ii = -1
    for s in range(2**Ns):
        config = int2bin(s, Ns)
        if np.count_nonzero(config) == Np:
            ii += 1 
            basis_dict[ii] = config
            invbasis_dict[int(bin2int(config))] = ii

    assert ii == dimH-1, "ii={}, dimH-1={}".format(ii, dimH-1)

    #print("basis_dict=", basis_dict)
    #print("invbasis_dict=", invbasis_dict)
    assert(np.all([ invbasis_dict[int(bin2int(basis_dict[ii]))] == ii for ii in range(dimH) ]))


    # build the Hamiltonian 
    Hamiltonian_tV = np.zeros((dimH, dimH))


    # kinetic term
    H_kin = np.zeros((dimH, dimH))
    for s1 in range(dimH):
        I1 = bin2int(basis_dict[s1])
        hop_from_to, I_primes, matrix_elems = kinetic_term2(int(I1), lattice) 
        #print("s1=", s1, "basis state=", basis_dict[s1])
        #print("I_s1=", bin2int(basis_dict[s1]))
        for (I2, me) in zip(I_primes, matrix_elems):
            s2 = invbasis_dict[I2]
            #print("s2=", s2, int2bin(I2, Ns))
            #print("me=", me)
            H_kin[s1, s2] =  t_hop * me 
           
    #print("H_kin:")
    #print(H_kin)

    # interaction term 
    H_int = np.zeros((dimH, dimH))
    for ii in range(dimH):
        config = basis_dict[ii]
        config_2D = config.reshape((lattice.nx, lattice.ny))
        Enn_int = 0.0
        for nd in range(lattice.coord // 2):
            Enn_int += ( np.roll(config_2D, shift=-1, axis=nd) * config_2D ).sum() 
        #ww = V_nnint * (np.roll(config, shift=-1) * config).sum(axis=-1)
        H_int[ii,ii] = V_nnint * Enn_int

    Hamiltonian_tV = H_kin + H_int

    #print("Hamiltonian=", Hamiltonian_tV)

    ###print("diagonalizing Hamiltonian")
    vals, vecs = np.linalg.eigh(Hamiltonian_tV)

    #print("unsorted energies=", vals)
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    print("energies=", vals[0:10])
    E_gs = np.min(vals)
    #print("ground state energy =", E_gs)
    print(V_nnint, E_gs)
    #print("ground state = ", vecs[:, 0])

    # measurements on the ground state 
    GS = vecs[:, 0]
    deg = 6 # Enter the ground state degeneracy here ! 
    szsz_corr = np.zeros(Ns)
    corr_ = np.zeros(Ns)

    for ii in range(dimH):
        config = basis_dict[ii]
        config_sz = 2*config - 1
        corr_[:] = 0.0
        for k in range(0, Ns):
            corr_[k] = (np.roll(config_sz, shift=-k) * config_sz).sum(axis=-1) / Ns
        ww = sum([ abs(vecs[ii, dd])**2  for dd in range(deg) ]) / float(deg)
        szsz_corr[:] += corr_[:] * ww # CAREFUL if ground state is degenerate ! Sum over degenerate states. 

    np.savetxt('ED_szsz_corr_Nx%dNy%dNp%dV%4.4f.dat' % (Nx, Ny, Np, V_nnint), szsz_corr[:, None])

    #plt.plot(range(Ns), szsz_corr[:], '--r')
    #plt.show()

