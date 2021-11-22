"""Exact diagonalization of the 1d t-V model (spin-polarized fermions)"""

import numpy as np
from bitcoding import *
from Slater_Jastrow_simple import ( kinetic_term, 
    Lattice1d )
from scipy.special import binom 

import matplotlib.pyplot as plt 

Np = 3#5
Ns = 7 # 13

dimH = int(binom(Ns, Np))
print("dimH=", dimH)
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

#print("basis_dict=", basis_dict)
#print("invbasis_dict=", invbasis_dict)
assert(np.all([ invbasis_dict[bin2int(basis_dict[ii]).item()] == ii for ii in range(dimH) ]))


# build the Hamiltonian 
Hamiltonian_tV = np.zeros((dimH, dimH))

t_par = 1.0 # t_par > 0, since kinetic_term() provides already a minus sign 
V_par = 3.0

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

#print("Hamiltonian=", Hamiltonian_tV)

print("diagonalizing Hamiltonian")
vals, vecs = np.linalg.eigh(Hamiltonian_tV)

#print("unsorted energies=", vals)
idx = np.argsort(vals)
vals = vals[idx]
vecs = vecs[:, idx]
print("energies=", vals[0:10])
print("ground state energy =", np.min(vals))
#print("ground state = ", vecs[:, 0])

# measurements on the ground state 
GS = vecs[:, 0]
szsz_corr = np.zeros(Ns)
corr_ = np.zeros(Ns)

for ii in range(dimH):
    config = basis_dict[ii]
    config_sz = 2*config - 1
    corr_[:] = 0.0
    for k in range(0, Ns):
        corr_[k] = (np.roll(config_sz, shift=-k) * config_sz).sum(axis=-1) / Ns
    szsz_corr[:] += corr_[:] * abs(GS[ii])**2

np.savetxt('ED_szsz_corr_Ns%dNp%dV%4.4f.dat' % (Ns, Np, V_par), szsz_corr[:, None])

plt.plot(range(Ns), szsz_corr[:], '--r')
plt.show()

