"""Read network parameters from checkpoint file and sample from the (hopefully converged) ansatz."""
import numpy as np
import torch 
import matplotlib.pyplot as plt 

from utils import default_dtype_torch 
torch.set_default_dtype(default_dtype_torch)

from VMC_common import VMCKernel, PhysicalSystem
#from slater_sampler_ordered_memory_layout import SlaterDetSampler_ordered
from slater_sampler_ordered import SlaterDetSampler_ordered
from HF import prepare_test_system_zeroT, HartreeFock_tVmodel

from graph_distance import * 


Nx = 4  # 15
Ny = 4
L=Nx

if L==6:
    S, num_dist = graph_distance_L6()
elif L==4: 
    S, num_dist = graph_distance_L4()

Nsites = Nx*Ny  # 15  # Nsites = 64 => program killed because it is using too much memory
space_dim = 2
Nparticles = 7
Vint = 1.0

num_samples = 200

phys_system = PhysicalSystem(nx=Nx, ny=Ny, ns=Nsites, num_particles=Nparticles, dim=space_dim, Vint=Vint)

# Aggregation of MADE neural network as Jastrow factor 
# and Slater determinant sampler. 
(eigvals, eigvecs) = HartreeFock_tVmodel(phys_system, potential="none", max_iter=2)
np.savetxt("eigvecs.dat", eigvecs)
#(_, eigvecs) = prepare_test_system_zeroT(Nsites=Nsites, potential='none', HF=True, PBC=False, Nparticles=Nparticles, Vnnint=Vint)
Sdet_sampler = SlaterDetSampler_ordered(Nsites=Nsites, Nparticles=Nparticles, single_particle_eigfunc=eigvecs, naive_update=False, optimize_orbitals=False)
SJA = SlaterJastrow_ansatz(slater_sampler=Sdet_sampler, num_components=Nparticles, D=Nsites, net_depth=2)

VMC = VMCKernel(energy_loc=phys_system.local_energy, ansatz=SJA)
del SJA

ckpt_outfile = 'state_Nx{}Ny{}Np{}V{}.pt'.format(Nx, Ny, Nparticles, Vint)

szsz_corr = np.zeros(Nsites)
szsz_corr_2D = np.zeros((phys_system.nx, phys_system.ny))
corr_2D = np.zeros((phys_system.nx, phys_system.ny))
av_sz = np.zeros((phys_system.nx, phys_system.ny))
av_density = np.zeros((phys_system.nx, phys_system.ny))
corr_ = np.zeros(Nsites)
tmp1 = np.zeros((phys_system.nx, phys_system.ny))
tmp2 = np.zeros((phys_system.nx, phys_system.ny))

corr_graph = np.zeros(L+1)

print("Now sample from the converged ansatz")
state_checkpointed = torch.load(ckpt_outfile)
VMC.ansatz.load_state_dict(state_checkpointed['net'], strict=False)
for _ in range(num_samples):
    sample_unfolded, log_prob_sample = VMC.ansatz.sample_unfolded()
    config = occ_numbers_collapse(sample_unfolded, Nsites).numpy()
    print("config=", config) 
    config_sz = 2*config - 1
    corr_[:] = 0.0
    for k in range(0, Nsites):
        corr_[k] = (np.roll(config_sz, shift=-k) * config_sz).sum(axis=-1) / Nsites
    szsz_corr[:] += corr_[:]

    # # 2D spin-spin correlations (or, alternatively, density-density correlations)
    # config_2D = config.reshape((phys_system.nx, phys_system.ny))
        
    # corr_graph[:] += graph_distance_corr(L, S, num_dist, config_2D, av_n = Nparticles/Nsites)
    
    # config_2D_sz = 2*config_2D - 1
    # tmp1[:,:] = 0.0 
    # tmp2[:,:] = 0.0
    # for x in range(0, phys_system.nx):
    #     for y in range(0, phys_system.ny):
    #        tmp1[x, y] = (np.roll(np.roll(config_2D_sz, shift=-x, axis=0), shift=-y, axis=1) * config_2D_sz).sum() / phys_system.ns
    #        tmp2[x, y] = (np.roll(np.roll(config_2D, shift=-x, axis=0), shift=-y, axis=1) * config_2D).sum() / phys_system.ns
    # szsz_corr_2D[:,:] += tmp1[:,:]
    # corr_2D[:,:] += tmp2[:,:]

    # av_sz[:,:] += config_2D_sz[:,:]
    # av_density[:,:] += config_2D[:,:]

szsz_corr[:] /= num_samples
szsz_corr_2D[:,:] /= num_samples
corr_2D[:,:] /= num_samples

av_sz[:,:] /= num_samples
av_density[:,:] /= num_samples

# connected correlation functions 
szsz_corr_2D_con = szsz_corr_2D[:,:] - av_sz[:,:]*av_sz[:,:]
corr_2D_con = corr_2D[:,:] - av_density[:,:]*av_density[:,:]

# graph distance correlations
corr_graph /= num_samples 

print("corr_graph=", corr_graph)
np.savetxt("corr_graph_Nx{}Ny{}Np{}V{}.dat".format(Nx, Ny, Nparticles, Vint), corr_graph)

np.savetxt("szsz_corr_Nx{}Ny{}Np{}V{}.dat".format(Nx, Ny, Nparticles, Vint), szsz_corr)
np.savetxt("szsz_corr_2D_con_Nx{}Ny{}Np{}V{}.dat".format(Nx, Ny, Nparticles, Vint), szsz_corr_2D_con)
np.savetxt("corr_2D_con_Nx{}Ny{}Np{}V{}.dat".format(Nx, Ny, Nparticles, Vint), corr_2D_con)

#plt.plot(range(Nsites), corr_2D_con.flatten(), '--b')
#plt.plot(range(Nsites), szsz_corr_2D_con.flatten(), '-r')
#plt.plot(range(Nsites), corr_2D.flatten(), '.b')
#plt.plot(range(Nsites), szsz_corr, '-g')
#plt.show()
