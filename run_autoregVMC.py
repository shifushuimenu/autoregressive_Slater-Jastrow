# IMPROVE: Organize imports globally in a better way
#          when making a package with an  __init__.py file. 
import numpy as np
import torch 
from utils import default_dtype_torch 

from one_hot import occ_numbers_collapse
from monitoring_old import logger 
from VMC_common import PhysicalSystem, VMCKernel
from SlaterJastrow_ansatz import SlaterJastrow_ansatz
#from slater_sampler_ordered_memory_layout import SlaterDetSampler_ordered
from slater_sampler_ordered import SlaterDetSampler_ordered
from test_suite import HartreeFock_tVmodel

import itertools

from sr_preconditioner import SR_Preconditioner, Identity_Preconditioner
from train import train_SR, Trainer

from time import time 
import argparse 

Parallel = False 

if Parallel:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()
    master = 0
    print("MPI: proc %d of %d" % (MPI_rank, MPI_size))
else:
    MPI_rank = 0
    MPI_size = 1
    master = 0

torch.set_default_dtype(default_dtype_torch)
torch.autograd.set_detect_anomaly(True)


desc_str="VMC with autoregressive Slater-Jastrow ansatz for t-V model of spinless fermions"
parser = argparse.ArgumentParser(description=desc_str)
group = parser.add_argument_group('physics parameters')
group.add_argument('Lx', type=int, help='width of square lattice')
group.add_argument('Ly', type=int, help='height of square lattice')
group.add_argument('Np', metavar='N', type=int, help='number of particles')
group.add_argument('Vint', metavar='V/t', type=float, help='nearest-neighbour interaction (V/t > 0 is repulsive)')
group = parser.add_argument_group('network parameters')
group.add_argument('--net_depth', type=int, default=2, help="number of layers in the Jastrow network (default: at least two autoregressive layers)")
group.add_argument('--optimize_orbitals', type=bool, default=False, help="co-optimize orbitals of Slater determinant (default=False)")
group = parser.add_argument_group('training parameters')
group.add_argument('num_epochs', metavar='max_epochs', type=int, help="number of training iterations")
group.add_argument('num_samples', type=int, help="number of samples per iteration")
group.add_argument('num_meas_samples', type=int, help="number of samples in measurement phase")
group = parser.add_argument_group('optimizer parameters')
group.add_argument('--seed', type=int, default=0, help="random seed, 0 for randomization")
group.add_argument('--optimizer', choices=['mySGD', 'SGD', 'SR', 'Adam', 'RMSprop'], default='SR')
group.add_argument('--lr', type=float, default=0.2, help="learning rate for SGD and SR (default=0.2); Adam and RMSprop have different learning rates.")
group.add_argument('--lr_SD', type=float, default=0.02, help="separate learning rate for parameters of the Slater determinant (default=0.02)")
group.add_argument('--lr_schedule', choices=['ReduceLROnPlateau', 'CyclicLR'], default=None, help="use learning rate scheduler")
group.add_argument('--monitor_convergence', type=bool, default=False, help="store model parameters on disk at every optimization step (default=False)")
args = parser.parse_args()


use_cuda = False
# set random number seed
if not args.seed or args.seed == 0:
    args.seed = np.random.randint(1, 10**8) + MPI_rank
torch.manual_seed(args.seed) 
np.random.seed(args.seed)
if use_cuda: torch.cuda.manual_seed_all(args.seed)

Lx = args.Lx # 5  # 15
Ly = args.Ly # 5
Nparticles = args.Np # 12
Vint = args.Vint #  Vint_array[MPI_rank]
num_epochs = args.num_epochs # 10 #1000 
num_samples = args.num_samples # 10 # 100  # samples per batch
num_bin = num_samples // 2 # 50 
num_meas_samples = args.num_meas_samples
optimizer_name = args.optimizer
optimize_orbitals = args.optimize_orbitals  # whether to include columns of P-matrix in optimization
lr = args.lr
lr_SD = args.lr_SD
lr_schedule = args.lr_schedule
print("args.lr_schedule=", args.lr_schedule)
monitor_convergence = args.monitor_convergence 

Nsites = Lx*Ly  # 15  # Nsites = 64 => program killed because it is using too much memory
space_dim = 2
paramstr = "Lx{}Ly{}Np{}V{}_{}".format(Lx, Ly, Nparticles, Vint, optimizer_name)
logger.info_refstate.outfile = "lowrank_stats_"+paramstr+".dat"

# for debugging:
# If deactivate_Jastrow == True, samples are drawn from the Slater determinant without the Jastrow factor. 
deactivate_Jastrow = False


        
# visualize the loss history
energy_list, precision_list = [], []
av_list, sigma_list = [], []
def _update_curve(energy, precision):
    energy_list.append(energy)
    precision_list.append(precision)
    Nb = len(energy_list)
    av = np.sum(energy_list) / Nb
    sigma = np.sqrt((np.sum([s**2 for s in precision_list]) / Nb) / Nb)
    av_list.append(av)
    sigma_list.append(sigma)
    if len(energy_list)%(num_epochs-1) == 0:
        xvals = np.arange(1, len(energy_list) + 1)

    MM = np.hstack((np.array(energy_list)[:,None], np.array(precision_list)[:,None],
                    np.array(av_list)[:,None], np.array(sigma_list)[:,None]))
    np.savetxt("energies_"+paramstr+".dat", MM)


ckpt_outfile = "state_"+paramstr+".pt"
def _checkpoint(VMCmodel):
    """Save most recent SJA state to disk."""
    state = {
        "energy": energy,
        "precision": precision, 
        "net": VMCmodel.ansatz.state_dict()
    }
    torch.save(state, ckpt_outfile)


phys_system = PhysicalSystem(nx=Lx, ny=Ly, ns=Nsites, num_particles=Nparticles, D=space_dim, Vint=Vint)

# Aggregation of MADE neural network as Jastrow factor 
# and Slater determi
with open("HF_energy_"+paramstr+".dat", "w") as fh:
    (eigvals, eigvecs) = HartreeFock_tVmodel(phys_system, potential="none", outfile=fh, max_iter=20)
np.savetxt("eigvecs"+paramstr+".dat", eigvecs)

Sdet_sampler = SlaterDetSampler_ordered(
        Nsites=Nsites, 
        Nparticles=Nparticles, 
        single_particle_eigfunc=eigvecs, 
        eigvals=eigvals, 
        naive_update=False, 
        optimize_orbitals=optimize_orbitals
        )

SJA = SlaterJastrow_ansatz(slater_sampler=Sdet_sampler, 
        num_components=Nparticles, 
        D=Nsites, 
        net_depth=args.net_depth, 
        deactivate_Jastrow=deactivate_Jastrow
        )

VMC = VMCKernel(energy_loc=phys_system.local_energy, ansatz=SJA)
del SJA

if optimizer_name in ['SR']:
    t1 = time()
    SR = SR_Preconditioner(num_params=sum([np.prod(p.size()) for p in VMC.ansatz.parameters()]), num_samples=num_samples, eps1=0.001, eps2=1e-6)
    t2 = time()
    VMC.t_SR += (t2-t1)
elif optimizer_name in ['mySGD']:
    SR = Identity_Preconditioner() # dummy class, just passes unmodified gradients through 
elif optimizer_name in ['SGD', 'Adam', 'RMSprop']:
    my_trainer = Trainer(VMC, lr, lr_schedule, optimizer_name, num_samples, num_bin, clip_local_energy=3.0, use_cuda=False)


E_exact = -3.6785841210741 #-3.86925667 # 0.4365456400025272 #-3.248988339062832 # -2.9774135797163597 #-3.3478904193465335

t0 = time()
t0_tmp = t0

# list of learning rates (for monitoring)
lrs = []

for i in range(num_epochs):

    if optimizer_name in ['mySGD', 'SR']:
        (energy, precision) = train_SR(VMC, learning_rate=lr, learning_rate_SD=lr_SD, num_samples=num_samples, num_bin=num_bin, use_cuda = use_cuda, precond=SR)
    else:
        (energy, precision) = my_trainer.train_standard_optimizer(lrs)

    t1_tmp = time()
    print('Step %d, dE/|E| = %.4f, elapsed = %.4f' % (i, -(energy - E_exact)/E_exact, t1_tmp-t0_tmp))
    _update_curve(energy, precision)
    _checkpoint(VMC)
    t0_tmp = time()

    print("monitor_convergence=", monitor_convergence)
    if monitor_convergence:
        # save model parameters in order to monitor convergence 
        with open("convergence_params_SD_"+paramstr+".dat", "a") as fh:
            for name, param in VMC.ansatz.named_parameters():
                if name in ['slater_sampler.T']:
                    arr = param.data.numpy().flatten()
                    fh.write( ("%16.10f " * arr.size + "\n") % (tuple(arr)) )

        ## save model parameters in order to monitor convergence 
        #with open("convergence_params_Jastrow_net0"+paramstr+".dat", "a") as fh:
        #    for name, param in VMC.ansatz.named_parameters():
        #        if name in ['net.0.weight']:
        #            arr = param.data.numpy().flatten()
        #            fh.write( ("%16.10f " * arr.size + "\n") % (tuple(arr)) )

    # remove
    np.savetxt("lrs"+paramstr+".dat", np.array(lrs))
    # remove


t1 = time()
with open("timings"+paramstr+".dat", "w") as fh:
    print("## Timings:", file=fh)
    print("## elapsed =%10.6f for %d samples per epochs and %d epochs" % (t1-t0, num_samples, num_epochs), file=fh)
    print("## t_psiloc=", VMC.t_psiloc, file=fh)
    print("## t_logprob_B=", VMC.ansatz.t_logprob_B, file=fh)
    print("## t_logprob_F=", VMC.ansatz.t_logprob_F, file=fh)
    print("## t_sampling=", VMC.t_sampling, file=fh)
    print("## t_locE=", VMC.t_locE, file=fh)
    print("## t_backward=", VMC.t_grads, file=fh)
    print("## t_stochastic_reconfiguration=", VMC.t_SR, file=fh)
    print("## total time det Schur complement=", VMC.ansatz.slater_sampler.t_det_Schur_complement, file=fh)
    print("## t_npix_=", VMC.ansatz.slater_sampler.t_npix_, file=fh)
    print("## t_linstorage=", VMC.ansatz.slater_sampler.t_linstorage, file=fh)
    print("## t_get_cond_prob=", VMC.ansatz.slater_sampler.t_get_cond_prob, file=fh)
    print("## t_update_state=", VMC.ansatz.slater_sampler.t_update_state, file=fh)
    print("## t_lowrank_linalg=",VMC.ansatz.slater_sampler.t_lowrank_linalg, file=fh)
    print("## t_gemm=",VMC.ansatz.slater_sampler.t_gemm, file=fh)


szsz_corr = np.zeros(Nsites)
szsz_corr_2D = np.zeros((phys_system.nx, phys_system.ny))
corr_ = np.zeros(Nsites)
corr_2D_ = np.zeros((phys_system.nx, phys_system.ny))


# ====================================================================================
print("Now sample from the converged ansatz")
###### setting up user-defined symmetry transformations for 2d lattice ######
s = np.arange(Nsites) # sites [0,1,2,....]
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


state_checkpointed = torch.load(ckpt_outfile)
VMC.ansatz.load_state_dict(state_checkpointed['net'], strict=True)
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
energy_list = np.zeros((num_meas_samples, 1))
#
t_sample = 0
for ii in range(num_meas_samples):
    t1 = time() 
    with torch.no_grad():
        sample_unfolded, log_prob_sample = VMC.ansatz.sample_unfolded()
    t2 = time()
    t_sample += (t2 - t1)
    config = occ_numbers_collapse(sample_unfolded, Nsites).squeeze().numpy()
    config_sz = 2*config - 1
#
    # local energy 
    ene, _ = VMC.local_measure([config], log_prob_sample)
    print("ene=", ene)
    energy_list[ii] = ene
    energy_av += ene
    energy2_av += ene**2 
#
    for tx, ty in itertools.product(range(Lx), range(Ly)):
        pair_list = [[i, translate(translate(i, tx, T_x), ty, T_y)] for i in range(Nsites)]
        ss1 = sum([config_sz[i] * config_sz[j] for (i,j) in pair_list]) / Nsites
        SzSzcorr[tx, ty] += ss1 
        ss2 = sum([config[i] * config[j] for (i,j) in pair_list]) / Nsites
        nncorr[tx, ty] += ss2 
#
        SzSzcorr2[tx, ty] += ss1**2 
        nncorr2[tx, ty] += ss2**2
#
SzSzcorr[:,:] /= num_meas_samples
nncorr[:,:] /= num_meas_samples 
SzSzcorr2[:,:] /= num_meas_samples 
nncorr2[:,:] /= num_meas_samples
# Central Limit Theorem for uncorrelated samples 
err_SzSzcorr = np.sqrt(SzSzcorr2[:,:] - SzSzcorr[:,:]**2) / np.sqrt(num_meas_samples)
err_nncorr = np.sqrt(nncorr2[:,:] - nncorr[:,:]**2) / np.sqrt(num_meas_samples)
#
np.savetxt("SzSzcorr_VMC_"+paramstr+".dat", SzSzcorr)
np.savetxt("err_SzSzcorr_VMC_"+paramstr+".dat", err_SzSzcorr)
np.savetxt("nncorr_VMC_"+paramstr+".dat", nncorr)
# Store also the connected density-density correlation function (valid only for translationally invariant systems)
##np.savetxt("nncorr_conn_VMC_"+paramstr+".dat", nncorr[:,:] - (Np / Nsites)**2 )
np.savetxt("err_nncorr_VMC_"+paramstr+".dat", err_nncorr)
#
energy_av /= num_meas_samples 
energy2_av /= num_meas_samples 
#
err_energy = np.sqrt(energy2_av - energy_av**2) / np.sqrt(num_meas_samples)
with open("energy_VMC_"+paramstr+".dat", "w") as fh:
    fh.write("energy = %16.10f +/- %16.10f" % (energy_av, err_energy))
# store timeseries => make histogram of non-Gaussian statistics 
np.savetxt("energy_TS_"+paramstr+".dat", energy_list)
#
print("## %d samples in %f seconds" % ( num_meas_samples, t_sample))
