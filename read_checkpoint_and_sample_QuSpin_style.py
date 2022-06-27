"""Read network parameters from checkpoint file and sample from the (hopefully converged) ansatz."""
import numpy as np
import torch 
import matplotlib.pyplot as plt 
import itertools

from utils import default_dtype_torch 
torch.set_default_dtype(default_dtype_torch)

from Slater_Jastrow_simple import *
#from slater_sampler_ordered_memory_layout import SlaterDetSampler_ordered
from slater_sampler_ordered import SlaterDetSampler_ordered
from test_suite import prepare_test_system_zeroT, HartreeFock_tVmodel



###### define model parameters ######
Lx, Ly = 6, 6 # linear dimension of spin 1 2d lattice
N_2d = Lx*Ly # number of sites
Np = 9  # number of particles 
#
J=1.0 # hopping matrix element
Vint=1.0 # nearest-neighbour interaction
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

num_samples = 2000

phys_system = PhysicalSystem(nx=Lx, ny=Ly, ns=N_2d, num_particles=Np, D=2, Vint=Vint)

# Aggregation of MADE neural network as Jastrow factor 
# and Slater determinant sampler. 
(eigvals, eigvecs) = HartreeFock_tVmodel(phys_system, potential="none", max_iter=2)
np.savetxt("eigvecs.dat", eigvecs)
#(_, eigvecs) = prepare_test_system_zeroT(N_2d=N_2d, potential='none', HF=True, PBC=False, Nparticles=Nparticles, Vnnint=Vint)
Sdet_sampler = SlaterDetSampler_ordered(Nsites=N_2d, Nparticles=Np, single_particle_eigfunc=eigvecs, eigvals=eigvals, naive_update=False, optimize_orbitals=False)
SJA = SlaterJastrow_ansatz(slater_sampler=Sdet_sampler, num_components=Np, D=N_2d, net_depth=2)

VMCmodel_ = VMCKernel(energy_loc=phys_system.local_energy, ansatz=SJA)
del SJA

ckpt_outfile = 'state_Nx{}Ny{}Np{}V{}.pt'.format(Lx, Ly, Np, Vint)

print("Now sample from the converged ansatz")
state_checkpointed = torch.load(ckpt_outfile)
VMCmodel_.ansatz.load_state_dict(state_checkpointed['net'], strict=False)

# init observables 
SzSzcorr = np.zeros((Lx, Ly))

for _ in range(num_samples):
    sample_unfolded, log_prob_sample = VMCmodel_.ansatz.sample_unfolded()
    config = occ_numbers_collapse(sample_unfolded, N_2d).squeeze().numpy()
    print("config=", config) 
    config_sz = 2*config - 1

    for tx, ty in itertools.product(range(Lx), range(Ly)):
        pair_list = [[i, translate(translate(i, tx, T_x), ty, T_y)] for i in range(N_2d)]
        SzSzcorr[tx, ty] += sum([config_sz[i] * config_sz[j] for (i,j) in pair_list]) / N_2d

SzSzcorr[:,:] /= num_samples

np.savetxt("SzSzcorr_VMC_"+paramstr+".dat", SzSzcorr)