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



sX, sY, sZ, sI = PauliMatrice()


state = 0.5*np.eye(2) + (1.0/(2.0*np.sqrt(2.0)))*(sX - sY)


magic = compute_magicSRE_fromRDM(state,1)
print('magic =', magic)

