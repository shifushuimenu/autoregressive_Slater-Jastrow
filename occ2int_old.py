def occ2int_spinless(occ_vector):
    """
    Map a spinless fermion occupation vector to an integer by interpreting 
    the occupation vector as the binary prepresentation of 
    an integer with the most significant bit to the right. 
    
        occ_vector = [1, 0, 1, 0]   ->   integer = 5
    """
    occ_vector = np.array(occ_vector, dtype=np.int8)
    s = 0
    for k in range(len(occ_vector)):
        # least significant bit to the right
        # if (occ_vector[-(k+1)] == 1):
        #     s = s + 2**k
        # least significant bit to the left            
        if (occ_vector[k] == 1):
            s = s + 2**k
    return s  


def occ2int_spinful(occ_vector_up, occ_vector_dn, debug=False):
    """
    Combine the occupation vectors for spin up and spin down 
    and map the resulting combined occupation vector to 
    an integer. The most significant bit is to the right.

    Example:
    ========
        occ_vector_up = [1, 0, 0, 1]
        occ_vector_dn = [0, 1, 1, 0]
        [occ_vector_up, occ_vector_dn] = [1, 0, 0, 1; 0, 1, 1, 0]  -> integer = 105
    """
    assert(len(occ_vector_up) == len(occ_vector_dn))
    occ_vector_up = np.array(occ_vector_up)
    occ_vector_dn = np.array(occ_vector_dn)
    occ_vector = np.hstack((occ_vector_up, occ_vector_dn))
    
    if (debug):
        print(occ_vector)

    return occ2int_spinless(occ_vector)


def int2occ_spinful(integer, Nsites):
    """
    Convert the integer representing an occupation number vector
    for spin up and spin down into a bitstring. 

    Example:
    ========
        occ_vector_up = [1, 0, 0, 1]
        occ_vector_dn = [0, 1, 1, 0]
        integer = 105 -> [occ_vector_up, occ_vector_dn] = [1, 0, 0, 1; 0, 1, 1, 0]     
    """
    Nspecies = 2

    # least significant bit to the right 
    i = integer 
    bitstring = []
    while(i != 0):
        bit = i % 2
        bitstring.insert(0, bit)
        i = i // 2
    # write leading zeros
    for _ in range(Nspecies*Nsites - len(bitstring)):
        bitstring.insert(0, 0)

    assert(len(bitstring) == 2*Nsites)

    return bitstring 
