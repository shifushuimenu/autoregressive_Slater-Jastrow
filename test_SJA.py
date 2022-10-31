
def init_test(N_particles=3, N_sites=5):
    """Initialize a minimal test system."""
    from test_suite import prepare_test_system_zeroT

    (N_sites, eigvecs) = prepare_test_system_zeroT(Nsites=N_sites)
    # Aggregation of MADE neural network as Jastrow factor 
    # and Slater determinant sampler. 
    Sdet_sampler = SlaterDetSampler(eigvecs, Nparticles=N_particles)
    SJA = SlaterJastrow_ansatz(slater_sampler=Sdet_sampler, num_components=N_particles, D=N_sites, net_depth=2)


def quick_tests():
    """Tests created ad hoc while debugging."""
    from test_suite import ( 
            prepare_test_system_zeroT
            )

    from bitcoding import bin2int
    from one_hot import occ_numbers_unfold 

    N_particles = 2
    N_sites = 5
    (N_sites, eigvecs) = prepare_test_system_zeroT(Nsites=N_sites)

    num_samples = 200

    # Aggregation of MADE neural network as Jastrow factor 
    # and Slater determinant sampler. 
    Sdet_sampler = SlaterDetSampler_ordered(Nsites=N_sites, Nparticles=N_particles, single_particle_eigfunc=eigvecs)
    SJA = SlaterJastrow_ansatz(slater_sampler=Sdet_sampler, num_components=N_particles, D=N_sites, net_depth=2)

    batch = torch.zeros(4, N_particles*N_sites)
    for i in range(4):
        sample = SJA.sample_unfolded()
        batch[i, ...] = sample[0]

    print("batch=", batch)
    print("SJA.psi_amplitude=", SJA.psi_amplitude(occ_numbers_collapse(batch, N_sites)))

    # SJA.log_prob() throws an error message. 
    print("SJA.log_prob=", SJA.log_prob(occ_numbers_collapse(batch, N_sites)))
    print("SDet.psi_amplitude=", Sdet_sampler.psi_amplitude(occ_numbers_collapse(batch, N_sites)))

    print("")
    print("test input of integer-coded states (batched):")
    print("=============================================")
    II = bin2int(occ_numbers_collapse(batch, N_sites))
    print("SDet.psi_amplitude_I=", Sdet_sampler.psi_amplitude_I(II))
    print("SJA.psi_amplitude_I=", SJA.psi_amplitude_I(II))

    print("")
    print("test behaviour w.r.t. introducing an additional batch dimension: ")
    print("================================================================ ")
    print("batch.shape=", batch.shape)
    print("batch.unsqueeze(dim=0).shape=", batch.unsqueeze(dim=0).shape)
    batch_unsq = batch.unsqueeze(dim=0)
    print("SJA.psi_amplitude(batch.unsqueeze(dim=0))=", SJA.psi_amplitude_unbatch(occ_numbers_collapse(batch_unsq, N_sites)))


def _test():
    import doctest
    doctest.testmod(verbose=False)
    print(quick_tests.__doc__)
    quick_tests()

if __name__ == '__main__':

    torch.set_default_dtype(default_dtype_torch)

    #_test()
    import matplotlib.pyplot as plt 
    from synthetic_data import * 
    from test_suite import ( 
            prepare_test_system_zeroT
            )

    from bitcoding import bin2int
    from one_hot import occ_numbers_unfold 

    N_particles = 3
    N_sites = 6
    (N_sites, eigvecs) = prepare_test_system_zeroT(Nsites=N_sites)

    num_samples = 1000

    # Aggregation of MADE neural network as Jastrow factor 
    # and Slater determinant sampler. 
    Sdet_sampler = SlaterDetSampler_ordered(Nsites=N_sites, Nparticles=N_particles, single_particle_eigfunc=eigvecs)
    SJA = SlaterJastrow_ansatz(slater_sampler=Sdet_sampler, num_components=N_particles, D=N_sites, net_depth=2)

    data_dim = 2**N_sites
    hist = torch.zeros(data_dim)
    model_probs = np.zeros(data_dim)
    model_probs2 = np.zeros(data_dim)
    DATA = Data_dist(Nsites=N_sites, Nparticles=N_particles, seed=678)

    batch = torch.zeros(num_samples, N_particles*N_sites)
    for i in range(num_samples):
        sample_unfolded, log_prob_sample = SJA.sample_unfolded() # returns one sample, but with batch dimension
        batch[i, ...] = sample_unfolded[0]
        s = occ_numbers_collapse(sample_unfolded, N_sites)
        print("s=", s)
        print("amplitude=", SJA.psi_amplitude(s))
        print("amp^2=", SJA.psi_amplitude(s)**2)
        print("prob=", SJA.prob(s))
        model_probs[DATA.bits2int(s.squeeze(dim=0))] = torch.exp(SJA.log_prob(s)).detach().numpy()
        model_probs2[DATA.bits2int(s.squeeze(dim=0))] = np.exp(log_prob_sample)
        s = s.squeeze(dim=0)
        hist[DATA.bits2int(s)] += 1
    hist /= num_samples

    f = plt.figure()
    ax0 = f.subplots(1,1)

    ax0.plot(range(len(hist)), np.array(hist), 'r--o', label="MADE samples hist")
    ax0.plot(range(len(model_probs)), np.array(model_probs), 'g--o', label="MADE: exp(log_prob)")
    ax0.plot(range(len(model_probs2)), np.array(model_probs2), 'b--o', label="MADE: prob_sampled")
    ax0.legend()

    plt.show()

    ##print("batch=", batch)
    ##print("SJA.psi_amplitude=", SJA.psi_amplitude(occ_numbers_collapse(batch, N_sites)))

    ##SJA.log_prob() throws an error message. 
    ##print("SJA.log_prob=", SJA.log_prob(occ_numbers_collapse(batch, N_sites)))
    ##print("SDet.psi_amplitude=", Sdet_sampler.psi_amplitude(occ_numbers_collapse(batch, N_sites)))

    ##print("")
    ##print("test input of integer-coded states (batched):")
    ##print("=============================================")
    ##II = bin2int(occ_numbers_collapse(batch, N_sites))
    ##print("SDet.psi_amplitude_I=", Sdet_sampler.psi_amplitude_I(II))
    ##print("SJA.psi_amplitude_I=", SJA.psi_amplitude_I(II))

    ##print("")
    ##print("test behaviour w.r.t. introducing an additional batch dimension: ")
    ##print("================================================================ ")
    ##print("batch.shape=", batch.shape)
    ##print("batch.unsqueeze(dim=0).shape=", batch.unsqueeze(dim=0).shape)
    ##batch_unsq = batch.unsqueeze(dim=0)
    ##print("SJA.psi_amplitude(batch.unsqueeze(dim=0))=", SJA.psi_amplitude_unbatch(occ_numbers_collapse(batch_unsq, N_sites)))

