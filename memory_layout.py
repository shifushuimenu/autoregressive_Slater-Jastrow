import numpy as np
import torch 
from profilehooks import profile

def store_G_linearly(G):
   assert len(G.shape) == 2 and G.shape[0] == G.shape[1]
   Ns = G.shape[0]
   g = np.array([G[0,0]])
   for ii in range(1, Ns):
      B = G[np.ix_(range(0, ii), range(ii, ii+1))].flatten()[:, None]
      C = G[np.ix_(range(ii, ii+1), range(0, ii))].flatten()[:, None]
      D = G[np.ix_(range(ii, ii+1), range(ii, ii+1))]
      
      g = np.vstack((g, B, C, D)).reshape(-1,1, order="F") # Fortran-like index order: "innermost index is fastest" 

   assert g.shape == (Ns*Ns, 1)
   return g 

def idx_linearly_stored_G(G_linear_mem, rows, cols, chunk, lr, lc):
    """
       `G_linear_mem` will be a torch tensor. 
      
       ________
       |      |  |
       |      |B |
       |      |  |
       -----------
       |__C___|D |
    """
    assert chunk in ("B", "C", "D")
    assert ( (len(rows) == 1 and len(cols) == 1)  or 
            (len(rows) > 1 and len(cols) == 1)  or 
            (len(rows) == 1 and len(cols) > 1)  )
    assert not (len(rows) == 1 and len(cols) > 1) or (rows[0] == cols[-1] + 1)
    assert not (len(cols) == 1 and len(rows) > 1) or (cols[0] == rows[-1] + 1)
    assert ( not (len(cols) == 1 and len(rows) == 1) or 
            (cols[0] == rows[0] or cols[0] == rows[0] + 1 or rows[0] == cols[0] + 1) )

    # avoid calling len(), instead provide this information explicitly when calling the function   
    assert lr == len(rows)
    assert lc == len(cols)

    a1 = max(rows[-1], cols[-1])

    if lr == 1:
        if lc == 1:
            if chunk == "D":
                assert rows[0] == cols[0]
                offset = a1*a1 + 2*a1
                return G_linear_mem[offset, 0][None, None] # D -> convert single element into matrix 
            elif chunk == "B":
                assert cols[0] > rows[0]
                offset = a1*a1
                return G_linear_mem[offset+rows[0], 0].reshape(-1,1) 
            elif chunk == "C":
                assert rows[0] > cols[0]
                offset = a1*a1 + a1 
                return G_linear_mem[offset+cols[0], 0].reshape(1,-1) # C -> reshape to row vector 
        else: # lc > 1 
            offset = a1*a1 + a1 + cols[0]
            return G_linear_mem[offset:offset+lc, 0].reshape(1,-1) # C -> reshape to row vector 
    else: # lr > 1 
        offset = a1*a1 + rows[0]
        return G_linear_mem[offset:offset+lr, 0].reshape(-1,1) # B -> reshape to column vector 


def idx_linearly_stored_G_blockB(G_linear_mem, rows, cols, chunk, lr, lc):
    """
        Block B is immediately adjacent to a square block A.
         _____________
        |       | |   |
        |   A   | | B |
        |_______|_|___|
        |       |     |
        |_______|_____|

    """
    assert chunk in ("B")
    assert rows[0] == 0
    assert rows[-1] + 1 == cols[0]
    
    # avoid calling len(), instead provide this information explicitly when calling the function   
    assert lr == len(rows)
    assert lc == len(cols)

    B_out = np.empty((lr, lc))

    offset = lr*lr
    B_out[:,0] = G_linear_mem[offset:offset+lr,0]
    for c in range(1, lc):
        stride = lr+2*c-1
        offset = offset + lr + stride
        B_out[:,c] = G_linear_mem[offset:offset+lr,0]

    return B_out 


def idx_linearly_stored_G_blockB1(G_linear_mem, rows, cols, chunk, lr, lc):
    assert chunk in ("B1")
    assert len(cols) == 1
    # avoid calling len(), instead provide this information explicitly when calling the function   
    assert lr == len(rows)
    assert lc == len(cols)

    col = cols[0]
    assert col > lr 
    i = (col-lr)
    offset = lr*lr + 2*lr*i + i*i
    return G_linear_mem[offset:offset+lr, 0].reshape(-1,1)




if __name__ == "__main__":
    from time import time 

    L = 6
    G = np.random.randn(L,L)
    G_lin = store_G_linearly(G)

    B1_lin = idx_linearly_stored_G(G_lin, [0,1,2], [3], "B")
    B1 = G[np.ix_([0,1,2], [3])]
    assert np.isclose(B1_lin, B1).all()

    C1_lin = idx_linearly_stored_G(G_lin, [3], [0,1,2], "C")
    C1 = G[np.ix_([3], [0,1,2])]
    assert np.isclose(C1_lin, C1).all()

    D1_lin = idx_linearly_stored_G(G_lin, [3], [3], "D")
    D1 = G[np.ix_([3], [3])]
    assert np.isclose(D1_lin, D1).all()


    # special cases: just one element array, but elements are different 
    B1_lin = idx_linearly_stored_G(G_lin, [1,2], [3], "B")
    B1 = G[np.ix_([1,2], [3])]
    assert np.isclose(B1_lin, B1).all()

    B_lin = idx_linearly_stored_G_blockB(G_lin, [0,1,2], [3,4,5], "B")
    B = G[np.ix_([0,1,2], [3,4,5])]
    assert np.isclose(B_lin, B).all()

    B_lin = idx_linearly_stored_G_blockB1(G_lin, [0,1,2], [5], "B1")
    B = G[np.ix_([0,1,2], [5])]
    assert np.isclose(B_lin, B).all()

    G = np.random.randn(L,L)
    G_lin = store_G_linearly(G)
    t0 = time()
    Ksites = []
    for i in range(L):
        if i==0:
            submat = idx_linearly_stored_G(G_lin, [i], [i], "D")
        else:
            submat = idx_linearly_stored_G(G_lin, Ksites, [i], "B")
            submat = idx_linearly_stored_G(G_lin, [i], Ksites, "C")
        Ksites.append(i)
    t1 = time()
    print("linear storage with indexing: elapsed =", t1-t0)

    t0 = time()
    Ksites = []
    for i in range(L):
        Ksites.append(i)
        if i==0:
            submat = G[Ksites[0], Ksites[0]]
        else: 
            A = submat 
            B = G[np.ix_(Ksites[:-1], [Ksites[-1]])] 
            C = G[np.ix_([Ksites[-1]], Ksites[:-1])]
            D = G[Ksites[-1], Ksites[-1]]
            #submat = np.block([[A, B],[C, D]])
    t1 = time()
    print("reuse submat: elapsed=", t1-t0)