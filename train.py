from Slater_Jastrow_simple import *
import time
import matplotlib.pyplot as plt

torch.set_default_dtype(default_dtype_torch)
# set random number seed
use_cuda = False
seed = 10086
torch.manual_seed(seed)
if use_cuda: torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
max_iter = 100
num_sites = 10
num_particles = 5

from test_suite import *
(nsites, U) = prepare_test_system_zeroT(Nsites=num_sites, potential='none')
print(U)

E_exact = 5.0


def train(model, learning_rate, use_cuda):
    '''
    train a model.

    Args:
        model (obj): a model that meet VMC model definition.
        learning_rate (float): the learning rate for SGD.
    '''
    initial_config = np.array([1, 1] * (model.ansatz.D // 2))
    if use_cuda:
        model.ansatz.cuda()

    while True:
        # get expectation values for energy, gradient and their product,
        # as well as the precision of energy.        
        with torch.no_grad():
            sample_list = model.ansatz.sample(initial_config, num_bath=40, num_sample=200)
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
    if len(energy_list)%10 == 0:
        plt.errorbar(np.arange(1, len(energy_list) + 1), energy_list, yerr=precision_list, capsize=3)
        # dashed line for exact energy
        plt.axhline(E_exact, ls='--')
        plt.show()

SJ_ansatz = SlaterJastrow(eigfunc=U, num_particles=num_particles)
model = VMCKernel(energy_loc=tVmodel_loc, ansatz=SJ_ansatz)

t0 = time.time()
for i, (energy, precision) in enumerate(train(model, learning_rate = 0.1, use_cuda = use_cuda)):
    t1 = time.time()
    print('Step %d, dE/|E| = %.4f, elapsed = %.4f' % (i, -(energy - E_exact)/E_exact, t1-t0))
    _update_curve(energy, precision)
    t0 = time.time()

    # stop condition
    if i >= max_iter:
        break