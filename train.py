from Slater_Jastrow_simple import *
import time
import matplotlib.pyplot as plt
from test_suite import prepare_test_system_zeroT

torch.set_default_dtype(default_dtype_torch)
# set random number seed
use_cuda = False
seed = 10086
torch.manual_seed(seed)
if use_cuda: torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
max_iter = 1000
Nsites = 13 # 10
Nparticles = 5 # 5
Vint = 5.0


def train(model, learning_rate, num_samples=10, use_cuda=False):
    '''
    train a model.

    Args:
        model (obj): a model that meet VMC model definition.
        learning_rate (float): the learning rate for SGD.
    '''
    if use_cuda:
        model.ansatz.cuda()

    while True:
        # get expectation values for energy, gradient and their product,
        # as well as the precision of energy.        
        sample_list = np.zeros((num_samples, Nsites)) 
        # Why not      with torch.no_grad():    during sampling ?
        for i in range(num_samples):
            sample_unfolded, sample_prob = SJA.sample_unfolded()
            sample_list[i] = occ_numbers_collapse(sample_unfolded, Nsites).numpy()
        energy, grad, energy_grad, precision = vmc_measure(model.local_measure, sample_list, num_bin=50)

        # update variables using steepest gradient descent
        g_list = [eg - energy * g for eg, g in zip(energy_grad, grad)]
        for var, g in zip(model.ansatz.parameters(), g_list):
            delta = learning_rate * g
            var.data -= delta
        yield energy, precision
        
        

# visualize the loss history
energy_list, precision_list = [], []
def _update_curve(energy, precision):
    energy_list.append(energy)
    precision_list.append(precision)
    if len(energy_list)%999 == 0:
        plt.errorbar(np.arange(1, len(energy_list) + 1), energy_list, yerr=precision_list, capsize=3, label="Slater-Jastrow")
        # dashed line for exact energy
        plt.axhline(E_exact, ls='--', label="exact")
        plt.title("$L$=%d, $N$=%d, $V/t$ = %4.4f" % (Nsites, Nparticles, Vint))
        plt.legend(loc="upper right")
        plt.show()

# Aggregation of MADE neural network as Jastrow factor 
# and Slater determinant sampler. 
(_, eigvecs) = prepare_test_system_zeroT(Nsites=Nsites, potential='none', HF=True, Nparticles=Nparticles, Vnnint=Vint)
Sdet_sampler = SlaterDetSampler_ordered(eigvecs, Nparticles=Nparticles)
SJA = SlaterJastrow_ansatz(slater_sampler=Sdet_sampler, num_components=Nparticles, D=Nsites, net_depth=2)

model = VMCKernel(energy_loc=tVmodel_loc, ansatz=SJA)

E_exact = -5.772442175744711

t0 = time.time()
for i, (energy, precision) in enumerate(train(model, learning_rate = 0.1, num_samples=100, use_cuda = use_cuda)):
    t1 = time.time()
    print('Step %d, dE/|E| = %.4f, elapsed = %.4f' % (i, -(energy - E_exact)/E_exact, t1-t0))
    _update_curve(energy, precision)
    t0 = time.time()

    # stop condition
    if i >= max_iter:
        break

# save converged ansatz
state = {
    "energy": energy,
    "precision": precision, 
    "net": SJA.state_dict()
}
torch.save(state, 'state_Ns{}Np{}V{}.pt'.format(Nsites, Nparticles, Vint))

print("Now sample from the converged ansatz")
for i in range(10):
    sample_unfolded, sample_prob = SJA.sample_unfolded()
    print(sample_prob, occ_numbers_collapse(sample_unfolded, Nsites).numpy())    