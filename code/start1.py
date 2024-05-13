"""

Numerical simulations of the non-stabilizing power of the Fourier Transform.


"""

import numpy as np
import randstab as rs
import sys
sys.path.append("../../dependencies")
from dependencies import *

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('../../dependencies/aps2column.mplstyle')

def FourierMatrix(n: int):
    """

    As it is defined in Mathematica and wikipedia the discrete (quantum) Fourier Transform.

    """

    matrix = np.zeros([2**n,2**n],dtype=complex)

    prefactor = 1.0/(np.sqrt(2**n))

    for i in range(0,2**n):
        for j in range(0,2**n):

            matrix[i,j] = prefactor*np.exp((2.0*np.pi*1j*i*j)/(2**n) )

    return matrix


Nsites = 5

FT = FourierMatrix(Nsites)

psi = np.zeros(2**Nsites,dtype=complex)
psi[0] = 1.0 + 0.0*1j
psi = psi/np.linalg.norm(psi)


magic               = compute_magicSRE(psi,Nsites)
entanglement, _     = vonNeumann(Nsites,int(Nsites/2.0), psi) 
print('magic =', magic, entanglement)

psi = CliffordOrbitBeyondLocal(psi, Nsites, 3)
psi = psi/np.linalg.norm(psi)

magic               = compute_magicSRE(FT @ psi,Nsites)
entanglement, _     = vonNeumann(Nsites,int(Nsites/2.0), FT @ psi) 
print('magic =', magic, entanglement)