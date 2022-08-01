"""Exact diagonalization of the 1d t-V model (spin-polarized fermions)"""

import numpy as np
from bitcoding import *
#from Slater_Jastrow_simple import ( kinetic_term2, 
#    Lattice1d, Lattice_rectangular )
from physics import (kinetic_term2, Lattice1d, Lattice_rectangular)
from scipy.special import binom 

import matplotlib.pyplot as plt 
import itertools


###### define model parameters ######
Lx, Ly = 3, 3 # linear dimension of spin 1 2d lattice
N_2d = Lx*Ly # number of sites
Np = 4  # number of particles 
#
J=1.0 # hopping matrix element
Vint=3.0 # nearest-neighbour interaction
paramstr = "Lx%dLy%dNp%dVint%f"%(Lx, Ly, Np, Vint)
#
###### setting up user-defined symmetry transformations for 2d lattice ######
s = np.arange(N_2d) # sites [0,1,2,....]
x = s%Lx # x positions for sites
y = s//Lx # y positions for sites
T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x +Lx*((y+1)%Ly) # translation along y-direction
P_x = x + Lx*(Ly-y-1) # reflection about x-axis
P_y = (Lx-x-1) + Lx*y # reflection about y-axis
#
def translate(s, n, T_d):
    """translate set of sites `s` using mapping `T_d` for n times"""
    for _ in range(n):
        s = T_d[s]
    return s


dimH = int(binom(N_2d, Np))
###print("dimH=", dimH)
#lattice = Lattice1d(ns=N_2d)
lattice = Lattice_rectangular(Lx, Ly)

# build the basis 
basis_dict = {}
invbasis_dict = {}
ii = -1
for s in range(2**N_2d):
    config = int2bin(s, N_2d)
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
        H_kin[s1, s2] =  1.0 * me 
        
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
    H_int[ii,ii] = Vint * Enn_int

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
print(Vint, E_gs)
#print("ground state = ", vecs[:, 0])

# measurements on the ground state 
GS = vecs[:, 0]
deg = 4 # Enter the ground state degeneracy here ! 
SzSzcorr = np.zeros((Lx,Ly))

for ii in range(dimH):
    config = basis_dict[ii]
    config_sz = 2*config - 1
    corr_ = np.zeros((Lx,Ly))
    for tx, ty in itertools.product(range(Lx), range(Ly)):
        pair_list = [[i, translate(translate(i, tx, T_x), ty, T_y)] for i in range(N_2d)]
        corr_[tx, ty] += sum([config_sz[i] * config_sz[j] for (i,j) in pair_list]) / N_2d
    ww = sum([ abs(vecs[ii, dd])**2  for dd in range(deg) ]) / float(deg)
    SzSzcorr[:,:] += corr_[:,:] * ww # CAREFUL if ground state is degenerate ! Sum over degenerate states. 

np.savetxt('ED_szsz_corr_Lx%dLy%dNp%dV%4.4f.dat' % (Lx, Ly, Np, Vint), SzSzcorr[:, :])

