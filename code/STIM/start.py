import stim
from typing import List
import sympy as sp

import sys

sys.path.append("../../dependencies")
from dependencies import *

n = 1

rotations: List[stim.Tableau] = []
for tab in stim.Tableau.iter_all(n):

    # tablueau ---> numerical matrix ---> symbolic matrix
    numerical_matrix = tab.to_unitary_matrix(endian="little")
    # approx_symboic_matrix = sp.Matrix(numerical_matrix)
    # print(approx_symboic_matrix)
    # symbolic_matrix = approx_symboic_matrix.applyfunc(sp.nsimplify)

    # print(symbolic_matrix)
    rotations.append(tab)
    print(tab)
    print(numerical_matrix)

print(40 * "-")
# print(rotations, len(rotations))


tableau = stim.Tableau.from_unitary_matrix(
    [
        [1, 0],
        [0, 1j],
    ],
    endian="little",
)
tableau = tableau * tableau
print(tableau)
unitary_matrix = tableau.to_unitary_matrix(endian="little")
print(unitary_matrix)

print(50 * "-")

n = 2

FT = FourierMatrix(n)
print(FT)
FTtableau = stim.Tableau.from_unitary_matrix(
    FT,
    endian="little",
)
