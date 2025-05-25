import stim
from typing import List
import numpy as np
import sys
import itertools
#from collections import Counter


import math

def cardinality_Cn_bar(n):
    power_term = 2 ** (n**2 + 2*n)
    product_term = math.prod([(4**j - 1) for j in range(1, n+1)])
    return power_term * product_term



def FourierMatrix(n: int):
    """

    As it is defined in Mathematica and wikipedia the discrete (quantum) Fourier Transform.

    """

    matrix = np.zeros([2**n, 2**n], dtype=complex)

    prefactor = 1.0 / (np.sqrt(2.0**n))

    for i in range(0, 2**n):
        for j in range(0, 2**n):

            matrix[i, j] = prefactor * np.exp((2.0 * np.pi * 1j * i * j) / (2.0**n))

    return matrix



def pauli_matrices_num():
    """
    Define the Pauli matrices using SymPy.

    Returns:

    """
    I = np.array([[1, 0], [0, 1]])  # Identity matrix
    X = np.array([[0, 1], [1, 0]])  # Pauli-X (sigma_x)
    Y = np.array([[0, -1j], [1j, 0]])  # Pauli-Y (sigma_y)
    Z = np.array([[1, 0], [0, -1]])  # Pauli-Z (sigma_z)

    return [I, X, Y, Z]




def compute_magic_numerically_from_unitary_num(matrix: np.ndarray, N: int):

    # define Pauli matrices
    lista = pauli_matrices_num()

    # basic computations
    gate_list = lista
    gate_list_id = [0,1,2,3]

    temp = [p for p in itertools.product(gate_list_id, repeat=N)]

    summ = 0
    for i in range(0,len(temp)):

        PauliString = gate_list[temp[i][0]]
        for j in range(1,len(temp[i])):
            PauliString = np.kron(PauliString,gate_list[temp[i][j]])

        lol = PauliString @ matrix  

        value = np.trace(lol) ** 4

        summ += value

    final = (1.0 / ((2.0)**N)) * summ

    return final.real






n = 4

# Example usage:
totCliffordUnitaries = cardinality_Cn_bar(n) 

print("Total number of Clifford unitaries:", totCliffordUnitaries)
# intial state trivial
matrix_zero = np.zeros([int(2**n), int(2**n)],dtype=complex)
matrix_zero[0, 0] = 1

# symbolic Fourier transform (DFT) matrix
FT = FourierMatrix(n)

# generate empty sympy matrix
matrix = np.zeros([int(2**n), int(2**n)],dtype=complex)

rotations: List[stim.Tableau] = []
counter = 1
magic = 0.0

unique_counts = {}

for tab in stim.Tableau.iter_all(n):

    if counter <  1000000:

        # 1. tablueau ---> numerical matrix
        numerical_Clifford = tab.to_unitary_matrix(endian="little")

        # 3. multiply out
        matrix = FT @ numerical_Clifford @ matrix_zero @ np.conjugate(np.transpose(numerical_Clifford)) @ np.conjugate(np.transpose(FT))

        # 4. compute the magic
        value_magic = compute_magic_numerically_from_unitary_num(matrix, n)

        value_magic = np.round(value_magic,10)

        unique_counts[value_magic] = unique_counts.get(value_magic, 0) + 1


        magic += -np.log2(value_magic)

        print("magic = ", counter,np.round(100*(counter/totCliffordUnitaries),2), value_magic,-np.log2(value_magic))
        counter += 1

    
    else:   
        break



print(30*'--')
counts_list = [[cnt, val] for val, cnt in unique_counts.items()]
print("Unique counts of value_magic:", counts_list)
print("Average magic:", magic/totCliffordUnitaries)