# L x L system with L=6
import numpy as np

def graph_distance_L6():
    L=6
    # num_dist[r] is the numbe rof points with graph distance r
    num_dist = [1, 4, 8, 10, 8, 4, 1]

    S = {}

    # origin 
    S[0] = ((0,0), )

    # all points with graph distance r = 1
    S[1] = ((0,1), (1,0), (5,0), (0, 5),)

    # all points with graph distance r = 2
    S[2] = ((0,2),(2,0),(1,1),(0,4),(4,0),(1,5),(5,1),(5,5),)

    # all points with graph distance r = 3
    S[3] = ((3,0),(0,3), (2,1), (1,2), (4,1), (1,4), (5,2), (2,5)
            , (5,4), (4,5),)

    # all points with graph distance r = 4
    S[4] = ((3,1), (1,3), (2,2), (2,4), (4,2), (5,3), (3,5), (4,4), )

    # all points with graph distance r = 5
    S[5] = ((2,3), (3,2), (3,4), (4,3),)

    # all points with graph distance r = 6
    S[6] = ((3,3),)

    for r in range(L+1):
        assert len(S[r]) == num_dist[r]
    
    return S, num_dist 
    
    
def graph_distance_L4():
    L=4
    num_dist = [1, 4, 6, 4, 1]
    S={}
    # origin 
    S[0] = ((0,0),)
    # all points with graph distance r = 1
    S[1] = ((0,1), (1,0), (3,0), (0,3),)
    # all points with graph distance r = 2
    S[2] = ((0,2), (2,0), (1,1), (3,1), (1,3), (3,3),)
    # all points with graph distance r = 3
    S[3] = ((2,1), (1,2), (2,3), (3,2),)
    # all points with graph distance r = 4
    S[4] = ((2,2),)
    
    for r in range(L+1):
        assert len(S[r]) == num_dist[r]
        
    return S, num_dist 
    
    
def graph_distance_corr(L, S, num_dist, config_2D, av_n):
    """Correlation function as a funtion of graph distance"""
    assert config_2D.shape == (L,L)
    config_2D = 2*config_2D - 1
    av = np.sum(config_2D) / L**2
    corr = np.zeros(L+1)
    for ix in range(L):
        config_2D_shiftx = np.roll(config_2D, shift=-ix, axis=0)
        for iy in range(L):
            config_2D_shifted = np.roll(config_2D_shiftx, shift=-iy, axis=1)                
            for r in (0,1,2,3,4,5,6,):                
                corr[r] += np.sum([(config_2D_shifted[(0, 0)] - av) * (config_2D_shifted[p] - av) for p in S[r]]) / (num_dist[r] * L**2)
                

    return corr

