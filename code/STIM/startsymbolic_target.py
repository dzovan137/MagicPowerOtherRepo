import stim
from typing import List
import sympy as sp
import numpy as np
import sys


def FourierMatrix(n: int):
    """

    As it is defined in Mathematica and wikipedia the discrete (quantum) Fourier Transform.

    """

    matrix = np.zeros([2**n, 2**n], dtype=complex)

    prefactor = 1.0 / (np.sqrt(2**n))

    for i in range(0, 2**n):
        for j in range(0, 2**n):

            matrix[i, j] = prefactor * np.exp((2.0 * np.pi * 1j * i * j) / (2**n))

    return matrix


def pauli_matrices():
    """
    Define the Pauli matrices using SymPy.

    Returns:

    """
    I = sp.Matrix([[1, 0], [0, 1]])  # Identity matrix
    X = sp.Matrix([[0, 1], [1, 0]])  # Pauli-X (sigma_x)
    Y = sp.Matrix([[0, -sp.I], [sp.I, 0]])  # Pauli-Y (sigma_y)
    Z = sp.Matrix([[1, 0], [0, -1]])  # Pauli-Z (sigma_z)

    return [I, X, Y, Z]


def convert_to_sympy_matrix(numpy_matrix, tolerance=1e-6):
    """
    Convert a NumPy matrix to a SymPy matrix by checking if real and imaginary parts
    of each entry are close to known symbolic constants.

    Args:
        numpy_matrix (np.ndarray): Input NumPy matrix with numerical values.
        tolerance (float): Tolerance for comparing numerical values to symbolic constants.

    Returns:
        sp.Matrix: A SymPy matrix with entries replaced by symbolic constants where applicable.
    """

    constants = [0, 0.5, 1, sp.sqrt(2), 1 / sp.sqrt(2), 1 / (2 * sp.sqrt(2)), 1 / 4, 1/8] 

    # Extend the list with negative versions of the constants
    constants.extend([-c for c in constants])

    # Create an empty SymPy matrix with the same shape as the input matrix
    sympy_matrix = sp.Matrix(
        numpy_matrix.shape[0], numpy_matrix.shape[1], lambda i, j: 0
    )

    # Iterate through the entries of the NumPy matrix
    for i in range(numpy_matrix.shape[0]):
        for j in range(numpy_matrix.shape[1]):
            entry = numpy_matrix[i, j]

            # Separate real and imaginary parts
            real_part = np.real(entry)
            imag_part = np.imag(entry)

            zz = 0
            # Check if the real and imaginary parts are close to any constant
            for constant in constants:

                # Check if the real part is close to the constant
                if abs(real_part - float(constant)) < tolerance:
                    real_part = sp.nsimplify(constant)

                # Check if the imaginary part is close to the constant
                if abs(imag_part - float(constant)) < tolerance:
                    imag_part = sp.nsimplify(constant)

                zz += 1

            # Reconstruct the entry with the simplified real and imaginary parts
            sympy_matrix[i, j] = real_part + sp.I * imag_part

    return sympy_matrix


def compute_magic_symbolic_from_unitary2(matrix):

    # define Pauli matrices
    lista = pauli_matrices()

    summ = 0
    for i in range(0, 4):
        for j in range(0, 4):

            # Reshape the tensor product result into a proper 2D array --- it contains a bit of AI code that helps to preform the tensor product which is non trivial with sympy newest versions
            lol_array = sp.Matrix(
                lista[i].rows * lista[j].rows,
                lista[i].cols * lista[j].cols,
                lambda r, c: lista[i][r // lista[j].rows, c // lista[j].cols]
                * lista[j][r % lista[j].rows, c % lista[j].cols],
            )
            lol = sp.Matrix(lol_array) * matrix  # Convert to SymPy Matrix

            value = (lol.trace()) ** 4

            summ += value

    final = -sp.log(sp.nsimplify((1 / 4) * summ), evaluate=False)

    return final


def compute_magic_symbolic_from_unitary3(matrix):

    # define Pauli matrices
    lista = pauli_matrices()

    summ = 0
    for i in range(0, 4):
        for j in range(0, 4):
            for k in range(0, 4):
                # Reshape the tensor product result into a proper 2D array
                lol_array = sp.Matrix(
                    lista[i].rows * lista[j].rows * lista[k].rows,
                    lista[i].cols * lista[j].cols * lista[k].cols,
                    lambda r, c: lista[i][
                        r // (lista[j].rows * lista[k].rows),
                        c // (lista[j].cols * lista[k].cols),
                    ]
                    * lista[j][
                        (r // lista[k].rows) % lista[j].rows,
                        (c // lista[k].cols) % lista[j].cols,
                    ]
                    * lista[k][r % lista[k].rows, c % lista[k].cols],
                )
                lol = sp.Matrix(lol_array) * matrix  # Convert to SymPy Matrix

                value = (lol.trace()) ** 4

                summ += value

    final = -sp.log(sp.nsimplify((1 / 8) * summ), evaluate=False)

    return final

def compute_magic_symbolic_from_unitary4(matrix):
    # define Pauli matrices
    lista = pauli_matrices()

    summ = 0
    # loop over all 4-tuples of Pauli indices
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    # build the 4-fold tensor product by explicit indexing
                    rows = lista[i].rows * lista[j].rows * lista[k].rows * lista[l].rows
                    cols = lista[i].cols * lista[j].cols * lista[k].cols * lista[l].cols
                    lol_array = sp.Matrix(
                        rows, cols,
                        lambda r, c: (
                            lista[i][
                                r // (lista[j].rows * lista[k].rows * lista[l].rows),
                                c // (lista[j].cols * lista[k].cols * lista[l].cols)
                            ]
                            * lista[j][
                                (r // (lista[k].rows * lista[l].rows)) % lista[j].rows,
                                (c // (lista[k].cols * lista[l].cols)) % lista[j].cols
                            ]
                            * lista[k][
                                (r // lista[l].rows) % lista[k].rows,
                                (c // lista[l].cols) % lista[k].cols
                            ]
                            * lista[l][
                                r % lista[l].rows,
                                c % lista[l].cols
                            ]
                        )
                    )
                    # multiply by the unitary and take the trace^4
                    lol = lol_array * matrix
                    value = (lol.trace())**4
                    summ += value

    # normalize by 2^4 = 16, then take -log
    final = -sp.log(sp.nsimplify((1/16) * summ), evaluate=False)
    return final



def compute_magic_symbolic_from_unitary(matrix, n):

    if n == 2:
        return compute_magic_symbolic_from_unitary2(matrix)
    elif n == 3:
        return compute_magic_symbolic_from_unitary3(matrix)
    elif n == 4:
        return compute_magic_symbolic_from_unitary4(matrix)
    else:
        raise ValueError("n must be 2 or 3")


n = 3

# intial state trivial
matrix_zero = sp.Matrix.zeros(2**n, 2**n)
matrix_zero[0, 0] = 1

# symbolic Fourier transform (DFT) matrix
FT = FourierMatrix(n)
FTsympy = convert_to_sympy_matrix(FT)
print("FTsympy = ", FTsympy)


# generate empty sympy matrix
matrix = sp.Matrix.zeros(2**n, 2**n)


rotations: List[stim.Tableau] = []
counter = 1
magic = 0
for tab in stim.Tableau.iter_all(n):

    if counter > 47745:
        
        # 1. tablueau ---> numerical matrix
        numerical_Clifford = tab.to_unitary_matrix(endian="little")

        # 2. numerical matrix ---> sympy matrix
        sympy_Clifford = convert_to_sympy_matrix(numerical_Clifford, tolerance=1e-4)

        # 3. multiply out
        matrix = FTsympy * sympy_Clifford * matrix_zero * sympy_Clifford.H * FTsympy.H

        # 4. compute the magic
        value_magic = compute_magic_symbolic_from_unitary(matrix, n)

        magic += value_magic

        print("magic = ", counter, value_magic)



    counter += 1

print(magic)
