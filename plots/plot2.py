"""

Numerical simulations of the non-stabilizing power of the Fourier Transform.


"""

# necessary libraries
import math

import numpy as np
import scipy.sparse as sparse
import scipy as sp

import sys
sys.path.append("../dependencies")
from dependencies import *

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('../dependencies/aps2column.mplstyle')

import pickle



var2 = 1      # 1 - computational basis, 2 - Clifford, 3 - Haar
var3 = 2      # 1 - Mlin, 2 - M2 
var4 = 100      # how many evaluations



xx = []
yya = []
yys = []

for var1 in range(2,10):


    filename = "../data/clusterdata/data_N" + str(var1) + "state_" + str(var2) + "method_" + str(var3) + "realizations_" + str(var4) + ".dat"


    # load the object from a file
    with open(filename, 'rb') as f:
        data_loaded = pickle.load(f)


    yyaverage   = data_loaded['yyaverage']
    yystd       = data_loaded['yystd'] 

    print(var1, yyaverage, yystd)

    xx.append(var1)
    yya.append(yyaverage)
    yys.append(yystd)





var2 = 2      # 1 - computational basis, 2 - Clifford, 3 - Haar
var3 = 2      # 1 - Mlin, 2 - M2 
var4 = 100      # how many evaluations



xxC = []
yyaC = []
yysC = []

for var1 in range(2,10):


    filename = "../data/clusterdata/data_N" + str(var1) + "state_" + str(var2) + "method_" + str(var3) + "realizations_" + str(var4) + ".dat"


    # load the object from a file
    with open(filename, 'rb') as f:
        data_loaded = pickle.load(f)


    yyaverage   = data_loaded['yyaverage']
    yystd       = data_loaded['yystd'] 

    print(var1, yyaverage, yystd)

    xxC.append(var1)
    yyaC.append(yyaverage)
    yysC.append(yystd)


var2 = 3      # 1 - computational basis, 2 - Clifford, 3 - Haar
var3 = 2      # 1 - Mlin, 2 - M2 
var4 = 100      # how many evaluations



xxH = []
yyaH = []
yysH = []

for var1 in range(2,10):


    filename = "../data/clusterdata/data_N" + str(var1) + "state_" + str(var2) + "method_" + str(var3) + "realizations_" + str(var4) + ".dat"


    # load the object from a file
    with open(filename, 'rb') as f:
        data_loaded = pickle.load(f)


    yyaverage   = data_loaded['yyaverage']
    yystd       = data_loaded['yystd'] 

    print(var1, yyaverage, yystd)

    xxH.append(var1)
    yyaH.append(yyaverage)
    yysH.append(yystd)





fig = plt.figure()


plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


grid = plt.GridSpec(1, 1, wspace=0.2, hspace=0.3)

ax = fig.add_subplot(grid[0, 0])

ax.errorbar(xx, yya, yys, fmt='o-', markersize=15, label=r"s = 1; computational basis state")
ax.errorbar(xxC, yyaC, yysC, fmt='s-', markersize=15, label=r"s = 2; Clifford state")
ax.errorbar(xxH, yyaH, yysH, fmt='^-', markersize=15, label=r"s = 3; Haar")



ax.legend(ncol=1,loc='best',fontsize='25',framealpha=1.0)

ax.set_xlabel(r' qubit number $N$',fontsize='35',labelpad=-2.0)
ax.set_ylabel(r' $ \mathbb{E}_{s} [M_{2} (U^{\rm FT} \vert \Psi_{s} \rangle ) ] $',fontsize='35',labelpad=5.0)


plt.savefig("plot2.png", dpi=300)


plt.show()

