# Hand-coded linear algebra routines for small systems 
import numpy as np

def my_det(A, m):
    """
        Hand-coded determinant formulae for m=1,2,3
        
        Input:
        A: square (m-times-m) torch tensor 
        m: int 
    """
    if m == 1:
        return A[0]
    elif m == 2:
        return A[0,0] * A[1,1] - A[1,0] * A[0,1]
    elif m == 3:
        # Sarrus formula 
        return (  A[0,0] * A[1,1] * A[2,2] + A[0,1] * A[1,2] * A[2,0] + A[0,2] * A[1,0] * A[2,1] 
                - A[2,0] * A[1,1] * A[0,2] - A[2,1] * A[1,2] * A[0,0] - A[2,2] * A[1,0] * A[0,1] 
                )
    elif m >= 4:
        return np.linalg.det(A)
        

