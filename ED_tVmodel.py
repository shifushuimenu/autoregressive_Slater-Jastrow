"""Exact diagonalization of the 1d t-V model (spin-polarized fermions)"""

import numpy as np
from bitcoding import *
from Slater_Jastrow_simple import ( kinetic_term, 
    Lattice1d )
from scipy.special import binom 


Np = 2 #5
Ns = 5 # 13

dimH = int(binom(Ns, Np))
lattice = Lattice1d(ns=Ns)

# build the basis 
basis_dict = {}
invbasis_dict = {}
ii = -1
for s in range(2**Ns):
    config = int2bin(s, Ns)
    if np.count_nonzero(config) == Np:
        ii += 1 
        basis_dict[ii] = config
        invbasis_dict[bin2int(config).item()] = ii

assert ii == dimH-1, "ii={}, dimH-1={}".format(ii, dimH-1)

print("basis_dict=", basis_dict)
print("invbasis_dict=", invbasis_dict)
assert(np.all([ invbasis_dict[bin2int(basis_dict[ii]).item()] == ii for ii in range(dimH) ]))


# build the Hamiltonian 
Hamiltonian_tV = np.zeros((dimH, dimH))

t_par = 1.0 # t_par > 0, since kinetic_term() provides already a minus sign 
V_par = 0.0

# kinetic term
H_kin = np.zeros((dimH, dimH))
for s1 in range(dimH):
    I1 = bin2int(basis_dict[s1]).item()
    _, I_primes, matrix_elems = kinetic_term([I1], lattice) # kinetic_term() requires batch dim
    # remove batch dimension
    I_primes = I_primes[0]; matrix_elems = matrix_elems[0]
    # filter non-zero matrix elements 
    idx = matrix_elems.nonzero()[0]
    matrix_elems = matrix_elems[idx]
    I_primes = I_primes[idx]
    for (I2, me) in zip(I_primes, matrix_elems):
        s2 = invbasis_dict[I2]
        H_kin[s1, s2] = t_par * me 
        
# interaction term 
H_int = np.zeros((dimH, dimH))
for ii in range(dimH):
    config = basis_dict[ii]
    ww = V_par * (np.roll(config, shift=-1) * config).sum(axis=-1)
    H_int[ii,ii] = ww

Hamiltonian_tV = H_kin + H_int

print("Hamiltonian=", Hamiltonian_tV)

vals, vecs = np.linalg.eig(Hamiltonian_tV)

idx = np.argsort(vals)
vals = vals[idx]
vecs = vecs[idx]
print("energies=", vals)
print("ground state energy =", np.min(vals))
print("ground state = ", vecs[0])




