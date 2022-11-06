import sys, os

testdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, testdir+'/..')


import unittest 

from selfMADE import selfMADE
from synthetic_data import Data_dist
from one_hot import occ_numbers_collapse 
from bitcoding import int2bin, bin2int
from utils import default_dtype_torch

import matplotlib.pyplot as plt 

import numpy as np
import torch 
torch.set_default_dtype(default_dtype_torch)

class TestMADE(unittest.TestCase):
    
    def test_fit_data_distribution(self, Nsites=6, Nparticles=2, visualize=False):
        """Tests created during debugging."""

        # synthetic data distribution 
        num_epochs = 160
        num_minibatches = 10
        minibatch_size = 100
        DATA = Data_dist(Nsites=Nsites, Nparticles=Nparticles, seed=678)
        data_entropy = DATA.entropy()
        data = DATA.generate_batch(batch_size=num_minibatches * minibatch_size, unfold=True)

        # Check that samples produced obey the distribution
        Nsamples=1280
        hist_data = torch.zeros(DATA.dim)
        hist_1st_pos = torch.zeros(Nsites)
        for i in range(Nsamples):
            sample = DATA.sample(rep='integer')
            bits = int2bin(sample, ns=Nsites)
            hist_1st_pos[bits.nonzero()[0][0]] += 1
            hist_data[sample] += 1
        hist_data /= Nsamples
        hist_1st_pos /= Nsamples

        # ==========================
        # fit synthetic data to MADE
        # ==========================
        model = selfMADE(D=Nsites, num_components=Nparticles, net_depth=2, bias_zeroth_component=hist_1st_pos)
        # number of trainable parameters (not taking masking of matrices into account)
        NUM_TRAINABLE_PARAMETERS = np.sum([np.prod(list(p.size())) for p in model.parameters()])
        #print("num. trainable params=", NUM_TRAINABLE_PARAMETERS)
        #for name, param in model.named_parameters():
        #    print(name, param)

        # set up optimizer: SGD works best and converges fastest 
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        #optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)

        torch.autograd.set_detect_anomaly(True)

        L=[]

        for epoch in range(num_epochs+1):
            for b in np.random.permutation(range(num_minibatches)):
                minibatch = data[b*minibatch_size:(b+1)*minibatch_size] #DATA.generate_batch(batch_size=minibatch_size, unfold=True)             
                optimizer.zero_grad()
                loss = model.cross_entropy(minibatch).mean()                        
                loss.backward()
                optimizer.step()
                L.append(loss.detach().numpy())
            scheduler.step()            
                
        is_ec_correctly_calculated = model.cross_entropy(data).mean().detach().numpy()
        is_ec_correctly_calculated2 = DATA.calc_entropy(hist_data)

        if visualize:
            f = plt.figure()
            ax0, ax00, ax1, ax2 = f.subplots(1,4)
            ax0.plot(range(len(DATA.probs)), np.array(DATA.probs), 'b-o', label="data dist")
            ax0.plot(range(len(hist_data)), np.array(hist_data), 'r--o', label="data histogram")
            ax0.legend()

            ax00.plot(range(Nsites), np.array(hist_1st_pos), 'g-o', label="hist 1st pos.")
            ax00.set_xlabel(r"position of 1st particle")
            ax00.set_ylabel(r"probability")
            ax00.legend()


            ax1.plot(range(len(L)), L, 'b-', label="loss")
            ax1.plot(range(len(L)), data_entropy * np.ones(len(L)), 'r--', label="entropy of data dist.")
            ax1.plot(range(len(L)), is_ec_correctly_calculated * np.ones(len(L)), 'g--', label="ce from full data dist")
            ax1.plot(range(len(L)), is_ec_correctly_calculated2 * np.ones(len(L)), 'y-', label="entropy from hist")
            ax1.legend()


        # Direct componentwise sampling from MADE
        # Now that we have trained the MADE network, we can start sampling from 
        # it and verify that it has learned the data distribution correctly.

        Nsamples=4080
        hist = torch.zeros(DATA.dim)
        model_probs = np.zeros(DATA.dim)
        with torch.no_grad():
            for i in range(Nsamples):
                sample_unfolded = model.sample_unfolded()        
                s = occ_numbers_collapse(sample_unfolded, Nsites)
                model_probs[np.array(bin2int(s.to(int).numpy()), dtype=np.int64)] = torch.exp(model.log_prob(s)).detach().numpy()
                s = s.squeeze(dim=0)
                hist[np.array(bin2int(s.to(int).numpy()), dtype=np.int64)] += 1
        hist /= Nsamples

        if visualize:
            ax2.plot(range(len(DATA.probs)), np.array(DATA.probs), 'b-o', label="data dist")
            ax2.plot(range(len(hist)), np.array(hist), 'r--o', label="MADE samples hist")
            ax2.plot(range(len(model_probs)), np.array(model_probs), 'g--o', label="MADE-predicted probs")
            ax2.legend()

            np.savetxt("DATA_probs.dat", DATA.probs)
            np.savetxt("model_probs.dat", model_probs)

            plt.show()

        assert np.all(np.isclose(sum(abs(np.array(DATA.probs) - np.array(model_probs))**2)/np.sqrt(DATA.dim), 0.0, atol=1e-3))
    
if __name__ == "__main__":
    unittest.main()
