from slater_evaluator import *
from test_suite import *

Ns, U = prepare_test_system_zeroT(5)

occ_vec = np.array([0,1,1,0,1])

SDeval = SlaterDetEvaluator(occ_vec, U[:,0:3])