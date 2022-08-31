import numpy as np
import itertools
import argparse
from time import time 
#
import torch 
from utils import default_dtype_torch 
torch.set_default_dtype(default_dtype_torch)
#
from one_hot import occ_numbers_collapse
from bitcoding import bin2pos
from Slater_Jastrow_simple import PhysicalSystem, VMCKernel
from SlaterJastrow_ansatz import SlaterJastrow_ansatz
from slater_sampler_ordered import SlaterDetSampler_ordered
from test_suite import HartreeFock_tVmodel
#
desc_str = "Read network parameters from checkpoint file and sample from the (hopefully converged) ansatz."
parser = argparse.ArgumentParser(description=desc_str)
parser.add_argument('Lx', type=int, help='width of square lattice')
parser.add_argument('Ly', type=int, help='height of square lattice')
parser.add_argument('Np', metavar='N', type=int, help='number of particles')
parser.add_argument('Vint', metavar='V/t', type=float, help='nearest neighbout interaction (V/t > 0 is repulsive)')
parser.add_argument('num_samples', type=int, help="number of samples")
#
args = parser.parse_args()
Lx = args.Lx; Ly = args.Ly; Np = args.Np; Vint = args.Vint; num_samples = args.num_samples 
#
J=1.0 # hopping matrix element
N_2d = Lx*Ly # number of sites
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
#
phys_system = PhysicalSystem(nx=Lx, ny=Ly, ns=N_2d, num_particles=Np, D=2, Vint=Vint)
#
# Aggregation of MADE neural network as Jastrow factor 
# and Slater determinant sampler. 
(eigvals, eigvecs) = HartreeFock_tVmodel(phys_system, potential="none", max_iter=2)
np.savetxt("eigvecs.dat", eigvecs)
#(_, eigvecs) = prepare_test_system_zeroT(N_2d=N_2d, potential='none', HF=True, PBC=False, Nparticles=Nparticles, Vnnint=Vint)
Sdet_sampler = SlaterDetSampler_ordered(Nsites=N_2d, Nparticles=Np, single_particle_eigfunc=eigvecs, eigvals=eigvals, naive_update=False, optimize_orbitals=True)
SJA = SlaterJastrow_ansatz(slater_sampler=Sdet_sampler, num_components=Np, D=N_2d, net_depth=2)
#
VMCmodel_ = VMCKernel(energy_loc=phys_system.local_energy, ansatz=SJA)
del SJA
#
# ===================
# Read checkpoint file 
# ===================
extension='_Adam'
fmt_string='state_Lx{}Ly{}Np{}V{}'+extension+'.pt'
ckpt_outfile = fmt_string.format(Lx, Ly, Np, Vint)
#
# ===================================================================
# Read columns of Hartree-Fock Slater determinant 
# (for comparing sign structure with co-optimized Slater determinant)
# ===================================================================
fmt_string = 'eigvecsLx{}Ly{}Np{}V{}'+extension+'.dat'
eigvecs_file = fmt_string.format(Lx, Ly, Np, Vint)
U_HF = np.loadtxt(eigvecs_file)
#
print("Now sample from the converged ansatz")
state_checkpointed = torch.load(ckpt_outfile)
VMCmodel_.ansatz.load_state_dict(state_checkpointed['net'], strict=True)
#
# init observables 
SzSzcorr = np.zeros((Lx, Ly))
nncorr = np.zeros((Lx, Ly))
#
# for calculating error bar
SzSzcorr2 = np.zeros((Lx, Ly))
nncorr2 = np.zeros((Lx, Ly))
#
energy_av = 0
energy2_av = 0
energy_list = np.zeros((num_samples, 1))
#
# average change of sign structure between optimized Slater determinant 
# (in presence of Jastrow) and Hartree-Fock Slater determinant
av_rel_sign = 0
#
t_sample = 0
for ii in range(num_samples):
    t1 = time() 
    with torch.no_grad():
        sample_unfolded, log_prob_sample = VMCmodel_.ansatz.sample_unfolded()
    t2 = time()
    t_sample += (t2 - t1)
    config = occ_numbers_collapse(sample_unfolded, N_2d).squeeze().numpy()
    config_sz = 2*config - 1
#
    # local energy 
    ene, _ = VMCmodel_.local_measure([config], log_prob_sample)
    print("sample nr.= ", ii, "ene=", ene)
    energy_list[ii] = ene
    energy_av += ene
    energy2_av += ene**2 
#
    for tx, ty in itertools.product(range(Lx), range(Ly)):
        pair_list = [[i, translate(translate(i, tx, T_x), ty, T_y)] for i in range(N_2d)]
        ss1 = sum([config_sz[i] * config_sz[j] for (i,j) in pair_list]) / N_2d
        SzSzcorr[tx, ty] += ss1 
        ss2 = sum([config[i] * config[j] for (i,j) in pair_list]) / N_2d
        nncorr[tx, ty] += ss2 
#
        SzSzcorr2[tx, ty] += ss1**2 
        nncorr2[tx, ty] += ss2**2
#
    # average sign change relative to Hartree-Fock Slater determinant 
    psi_coopt = VMCmodel_.ansatz.slater_sampler.psi_amplitude([config]).item()
    pos = bin2pos([config])
    submat = U_HF[pos, 0:Np]
    psi_HF = np.linalg.det(submat)
    av_rel_sign += np.sign(psi_coopt) * np.sign(psi_HF)
    print("psi_HF=", psi_HF, "psi_coopt=", psi_coopt)


SzSzcorr[:,:] /= num_samples
nncorr[:,:] /= num_samples 
SzSzcorr2[:,:] /= num_samples 
nncorr2[:,:] /= num_samples
err_SzSzcorr = np.sqrt(SzSzcorr2[:,:] - SzSzcorr[:,:]**2) / np.sqrt(num_samples)
err_nncorr = np.sqrt(nncorr2[:,:] - nncorr[:,:]**2) / np.sqrt(num_samples)
#
av_rel_sign /= num_samples 
np.savetxt("av_rel_sign_"+paramstr+".dat", np.array([av_rel_sign]))
#
np.savetxt("SzSzcorr_VMC_"+paramstr+".dat", SzSzcorr)
np.savetxt("err_SzSzcorr_VMC_"+paramstr+".dat", err_SzSzcorr)
np.savetxt("nncorr_VMC_"+paramstr+".dat", nncorr)
# Store also the connected density-density correlation function (valid only for translationally invariant systems)
##np.savetxt("nncorr_conn_VMC_"+paramstr+".dat", nncorr[:,:] - (Np / N_2d)**2 )
np.savetxt("err_nncorr_VMC_"+paramstr+".dat", err_nncorr)
#
energy_av /= num_samples 
energy2_av /= num_samples 
#
err_energy = np.sqrt(energy2_av - energy_av**2) / np.sqrt(num_samples)
with open("energy_VMC_"+paramstr+".dat", "w") as fh:
    fh.write("energy = %16.10f +/- %16.10f" % (energy_av, err_energy))
# store timeseries => make histogram of non-Gaussian statistics 
np.savetxt("energy_TS_"+paramstr+".dat", energy_list)
#
print("## %d samples in %f seconds" % ( num_samples, t_sample))
