"""

Numerical simulations of the non-stabilizing power of the Fourier Transform.


"""

# necessary libraries
import math

import numpy as np
import scipy.sparse as sparse
import scipy as sp

import randstab as rs
import sys
sys.path.append("../../dependencies")
from dependencies import *

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('../../dependencies/aps2column.mplstyle')





zero    = np.array([1.0,0.0])
one     = np.array([0.0,1.0])

xx = []
yy = []

for Nsites in range(2,7):

    basis_states = generate_basis_states(Nsites)
    FT = FourierMatrix(Nsites)

    tampon = []
    for state in basis_states:

        #state_vec = state[len(state)-1]

        if (state[len(state)-1] == 0):
            state_vec = zero
        else:
            state_vec = one

        for i in range(len(state)-2,-1,-1):

            if (state[i] == 0):
                state_vec = np.kron(zero,state_vec)
            if (state[i] == 1):
                state_vec = np.kron(one,state_vec)


        
        entanglement, _     = vonNeumann(Nsites,int(Nsites/2.0), FT @ state_vec) 
        magic   = compute_magicSRE(FT @ state_vec,Nsites)
        print(state, magic, entanglement)
        tampon.append(magic)
        print('-------------')

    xx.append(Nsites)
    yy.append(np.average(tampon))


print(xx)
print(yy)

# Fit the data to the linear function
params, covariance = sp.optimize.curve_fit(linear_func, xx, yy)

# Get the fitted parameters
m_fit, c_fit = params

# Print the fitted parameters
print("Fitted slope (m):", m_fit)
print("Fitted intercept (c):", c_fit)


xx_fit = []
yy_fit = []
for i in range(2,Nsites+4):
    xx_fit.append(i)
    yy_fit.append(c_fit + m_fit*i)



fig = plt.figure()
grid = plt.GridSpec(1, 1, wspace=0.2, hspace=0.3)

ax = fig.add_subplot(grid[0, 0])
ax.set_title(r"Checking the computational basis states. ",fontsize=20)

ax.scatter(xx,yy,marker='s',color='blue',label='direct magic ')
ax.plot(xx_fit,yy_fit,color='red',label=r'direct magic linear fit; $f(N) = a + b N, a = -0.76; b = 0.68$')


ax.legend(ncol=1,loc='best',fontsize='25',framealpha=0.5)

plt.savefig("test1.png", dpi=300)


plt.show()
