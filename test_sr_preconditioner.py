import unittest 
import numpy as np
import torch 
import torch.nn as nn

from utils import default_dtype_torch

from sr_preconditioner import SR_Preconditioner

from VMC_common import PhysicalSystem, VMCKernel
from train import vmc_measure 
from SlaterJastrow_ansatz import SlaterJastrow_ansatz
from slater_sampler_ordered import SlaterDetSampler_ordered
from HF import HartreeFock_tVmodel
from one_hot import occ_numbers_collapse

class TestInterface(unittest.TestCase):
    def test_flatten_grads_into_vector(self):
        # Build a simple neural net 
        nin=10; nout=10
        hiddens = [5,4,5]
        num_blocks = 2
        block = []
        hs = [nin] + hiddens + [nout]
        for h0, h1 in zip(hs[:], hs[1:]):
            block.extend([
                nn.Linear(in_features=h0, out_features=h1, bias=True), 
                nn.ReLU()                
            ])

        net = block * num_blocks + [nn.Sigmoid()]
        net = nn.Sequential(*net)

        x_in = torch.randn(nin)
        x_out = torch.sum(net(x_in))
        x_out.backward()

        grad = [p.grad.data for p in net.parameters()]

        num_params = np.sum([np.prod(p.size()) for p in net.parameters()])
        num_samples = 10
        SR = SR_Preconditioner(num_params=num_params, num_samples=num_samples)

        grad_flat = SR._flatten(grad)
        grad_new = SR._unflatten(grad_flat, grad)

        nested_booleans = [p1 == p2 for (p1,p2) in zip(grad, grad_new)
        l = True
        for t in nested_booleans:
            l = l and torch.all(t.flatten())
        assert l

class MinimalUsageExample(unittest.TestCase):

    def setUp(self):

        torch.set_default_dtype(default_dtype_torch)
        torch.autograd.set_detect_anomaly(True)

        # it's good to be reproducible
        seed=43
        torch.manual_seed(seed)
        np.random.seed(seed)

        Nx=4; Ny=4; Np=3
        self.Ns=Nx*Ny
        self.num_samples = 100

        phys_system = PhysicalSystem(nx=Nx, ny=Ny, ns=self.Ns, num_particles=Np, dim=2, Vint=3.0)
        (eigvals, eigvecs) = HartreeFock_tVmodel(phys_system, potential="none", max_iter=20)

        Sdet_sampler = SlaterDetSampler_ordered(
                Nsites=self.Ns, 
                Nparticles=Np, 
                single_particle_eigfunc=eigvecs, 
                eigvals=eigvals, 
                naive_update=False, 
                optimize_orbitals=True
                )
        SJA = SlaterJastrow_ansatz(slater_sampler=Sdet_sampler, 
                num_components=Np, 
                D=self.Ns, 
                net_depth=2
                )

        self.VMCmodel = VMCKernel(energy_loc=phys_system.local_energy, ansatz=SJA)
        num_params = sum([np.prod(p.size()) for p in self.VMCmodel.ansatz.parameters()])
        self.SR = SR_Preconditioner(num_params=num_params, num_samples=self.num_samples, eps1=1e-3, eps2=0.0)


    def test_joint_params_Jastrow_Sdet(self):

        learning_rate = 0.2
        num_epochs = 3

        for ii in range(num_epochs):
            sample_list = np.zeros((self.num_samples, self.Ns))
            log_probs = np.zeros((self.num_samples,))
            with torch.no_grad():
                for ii in range(self.num_samples):
                    x_out, log_prob = self.VMCmodel.ansatz.sample_unfolded()
                    log_probs[ii] = log_prob
                    sample_list[ii] = occ_numbers_collapse(x_out, self.Ns).numpy()

            av_H, av_Ok, av_HtimesOk, precision = vmc_measure(self.VMCmodel.local_measure, sample_list, log_probs, self.SR, num_bin=5)

            # gradient of the energy
            # grad_k = <H * Delta_k> - <H>*<Delta_k>
            grad_list = [av_HtimesOk - av_H*av_Ok for (av_HtimesOk, av_Ok) in zip(av_HtimesOk, av_Ok)]
            grad_list = self.SR.apply_Sinv(grad_list)

            for (name, par), g in zip(self.VMCmodel.ansatz.named_parameters(), grad_list):
                if name == 'slater_sampler.P':
                    print("grad SD.P=", g)
                delta = learning_rate * g
                par.data -= delta
            
            print("T=", self.VMCmodel.ansatz.slater_sampler.T.data)
            self.VMCmodel.ansatz.slater_sampler.rotate_orbitals()


if __name__ == "__main__":
    unittest.main()
