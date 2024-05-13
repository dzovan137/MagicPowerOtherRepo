"""

Numerical simulations of the non-stabilizing power of the Fourier Transform.

DOING THE QUANTUM FOURIER TRANSFORM CIRCUIT STEP BY STEP ! 


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

U1 = U1N3()
U2 = U2N3()
U3 = U3N3()
U4 = U4N3()
U5 = U5N3()
U6 = U6N3()
U7 = U7N3()

print(np.round(U7,4))



Nsites = 3

FT = FourierMatrix(Nsites)

tampon  = []
tampon1 = []
tampon2 = []
tampon3 = []
tampon4 = []
tampon5 = []
tampon6 = []
tampon7 = []

FT = FourierMatrix(Nsites)

for i in range(0,1000):

    psi = rs.random_stabilizer_state(Nsites)

    after   = compute_magicSRE(FT @ psi,Nsites)
    after1   = compute_magicSRE(U1 @ psi,Nsites)
    after2   = compute_magicSRE(U2 @ U1 @ psi,Nsites)
    after3   = compute_magicSRE(U2 @ U1 @ psi,Nsites)
    after4   = compute_magicSRE(U4 @ U2 @ U1 @ psi,Nsites)
    after5   = compute_magicSRE(U5 @ U4  @ U2 @ U1 @ psi,Nsites)
    after6   = compute_magicSRE(U6 @ U5 @ U4  @ U2 @ U1 @ psi,Nsites)
    after7   = compute_magicSRE(U7 @ U6 @ U5 @ U4 @ U2 @ U1 @ psi,Nsites)

    print(i,Nsites,after)

    tampon.append(after)
    tampon1.append(after1)
    tampon2.append(after2)
    tampon3.append(after3)
    tampon4.append(after4)
    tampon5.append(after5)
    tampon6.append(after6)
    tampon7.append(after7)

print('**************************')
print('after =',np.average(tampon))
print('after1 =',np.average(tampon1))
print('after2 =',np.average(tampon2))
print('after3 =',np.average(tampon3))
print('after4 =',np.average(tampon4))
print('after5 =',np.average(tampon5))
print('after6 =',np.average(tampon6))
print('after7 =',np.average(tampon7))

