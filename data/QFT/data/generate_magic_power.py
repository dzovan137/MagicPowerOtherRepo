"""

Code to generate data for the Fourier Transform stuff

"""

# necessary libraries
import math

import numpy as np
import scipy.sparse as sparse
import scipy as sp


import sys
sys.path.append("../dependencies")
from dependencies import *

import pickle



var1 = int(sys.argv[1])      # qubit number
var2 = int(sys.argv[2])      # 1 - computational basis, 2 - Clifford, 3 - Haar
var3 = int(sys.argv[3])      # 1 - Mlin, 2 - M2 
var4 = int(sys.argv[4])      # how many evaluations

N = var1

xx = []
yy = []

# generate the FT matrix unitary
FT = FourierMatrix(N)


# run over different realization of input states
for i in range(0,var4):


	# which state
    if (var2 == 1):

    	psi = RandomComputationalBasisState(N)

    elif(var2 == 2):

    	psi = random_stabilizer_state(N)

    elif(var2 == 3):

    	psi = GenerateHaarState(N)


    psi = FT @ psi



    # which thing to compute
    if(var3 == 1):

    	value  = Mlin(psi,N)

    elif(var3 == 2):

    	value  = compute_magicSRE(psi,N)
    

    print(i)
    xx.append(i)
    yy.append(value)






average = np.average(yy)
std = np.std(yy)

print('average =',average, 'std = ', std)

filename = "data_N" + str(var1) + "state_" + str(var2) + "method_" + str(var3) + "realizations_" + str(var4) + ".dat"
data = {'xx': xx,
        'yy': yy,
        'yyaverage': average,
        'yystd': std
        }


# save the object to a file
with open(filename, 'wb') as f:
    pickle.dump(data, f)




"""
pickle loading data


# load the object from a file
with open(filename, 'rb') as f:
    data_loaded = pickle.load(f)


std   = data_loaded['yystd']
print('loaded std', std)
"""

