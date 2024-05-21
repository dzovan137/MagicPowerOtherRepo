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
sys.path.append("../dependencies")
from dependencies import *

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('../dependencies/aps2column.mplstyle')

zero    = np.array([1.0,0.0])
one     = np.array([0.0,1.0])

T = (1.0/np.sqrt(2.0))*(zero + np.exp(1j*np.pi/4.0)*one)
print('TEST = ',compute_magicSRE(T,1))


Nsites = 4

FT = FourierMatrix(Nsites)

xx = []
yyHAAR = []
yyFT   = []

yylin = []

for Nsites in range(2,4):
    tamponHAAR = []
    tamponFT = []
    FT = FourierMatrix(Nsites)

    for i in range(0,100):

        psi = np.zeros(2**Nsites,dtype=complex)
        psi[0] = 1.0 + 0.0*1j
        #psi[1] = 1.0 + 0.0*1j
        psi = psi/np.linalg.norm(psi)
        #psi = CliffordOrbitBeyondLocal(psi, Nsites, 10)


        psi = HaarOrbit(psi,Nsites,1000)
        afterHAAR  = compute_magicSRE(psi,Nsites)
        tamponHAAR.append(afterHAAR)

        psi = rs.random_stabilizer_state(Nsites)
        afterFT   = compute_magicSRE(FT @ psi,Nsites)
        tamponFT.append(afterFT)

        print(Nsites, i)

    xx.append(Nsites)
    yyHAAR.append(np.average(tamponHAAR))
    yyFT.append(np.average(tamponFT))


    print('AVERAGE =', Nsites, np.average(tamponHAAR),np.average(tamponFT))



# Fit the data to the linear function
params, covariance = sp.optimize.curve_fit(linear_func, xx, yyHAAR)

# Get the fitted parameters
m_fitHAAR, c_fitHAAR = params

# Print the fitted parameters
print("HAAR Fitted slope (m):", m_fitHAAR)
print("HAAR Fitted intercept (c):", c_fitHAAR)
print('HAAR covariance =', covariance)



# Fit the data to the linear function
params, covariance = sp.optimize.curve_fit(linear_func, xx, yyFT)

# Get the fitted parameters
m_fitFT, c_fitFT = params

# Print the fitted parameters
print("FT Fitted slope (m):", m_fitFT)
print("FT Fitted intercept (c):", c_fitFT)
print('FT covariance =', covariance)






xx_fit = []
yy_fitHAAR = []
yy_fitFT = []
yy_max = []

yy_HAARlower = []
for i in range(2,Nsites+4):
    xx_fit.append(i)
    yy_fitHAAR.append(c_fitHAAR + m_fitHAAR*i)
    yy_fitFT.append(c_fitFT + m_fitFT*i)

    yy_max.append(np.log2(2**(i) + 1) - np.log2(2.0))

    yy_HAARlower.append(np.log2(2**i + 3.0) - 2.0*np.log2(2.0))

############################ BASIS STATES STUFF ##################
zero    = np.array([1.0,0.0])
one     = np.array([0.0,1.0])

xx = []
yy = []

for Nsites in range(2,4):

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


        magic   = compute_magicSRE(FT @ state_vec,Nsites)
        print(state, magic )
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
ax.set_title(r"  ",fontsize=20)

ax.scatter(xx,yyHAAR,marker='s',color='blue',label=' HAAR state ')
ax.plot(xx_fit,yy_HAARlower,color='blue',alpha=0.5, label=r'HAAR lower bound $\log_2{(2^N + 3)} - 2 \log_2{(2)}$')

ax.scatter(xx,yyFT,marker='s',color='green',label=' FT to STAB ')
ax.plot(xx_fit,yy_fitFT,color='green',label='y = %.2f x + %.2f' %( m_fitFT,c_fitFT))

ax.scatter(xx,yy,marker='o',color='green',label=' FT to STAB basis state ')
ax.plot(xx_fit,yy_fit,color='green',label='y = %.2f x + %.2f' %( m_fit,c_fit))



ax.plot(xx_fit,yy_max,color='red',label=r'max $M_2$ possible $\log_2{(2^N + 1)} - \log_2{(2)}$')



#ax.scatter(xx,yylin,marker='s',color='magenta',label=r'linearized magic $M_{\rm lin} = 1 - e^{-M_2}$')
#ax.plot(xx_fit,yy_linbound,color='black',label=r'bounding stuff')


ax.legend(ncol=1,loc='best',fontsize='20',framealpha=0.5)

plt.savefig("test.png", dpi=300)


plt.show()

