import stim
from typing import List
import numpy as np
import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def cardinality_Cn_bar(n):
    power_term = 2 ** (n**2 + 2*n)
    product_term = math.prod([(4**j - 1) for j in range(1, n+1)])
    return power_term * product_term


def FourierMatrix(n: int):
    matrix = np.zeros([2**n, 2**n], dtype=complex)
    prefactor = 1.0 / (np.sqrt(2.0**n))
    for i in range(0, 2**n):
        for j in range(0, 2**n):
            matrix[i, j] = prefactor * np.exp((2.0 * np.pi * 1j * i * j) / (2.0**n))
    return matrix


def pauli_matrices_num():
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    return [I, X, Y, Z]


def compute_magic_numerically_from_unitary_num(matrix: np.ndarray, N: int):
    lista = pauli_matrices_num()
    gate_list_id = [0,1,2,3]
    temp = [p for p in itertools.product(gate_list_id, repeat=N)]
    summ = 0
    for i in range(0, len(temp)):
        PauliString = lista[temp[i][0]]
        for j in range(1, len(temp[i])):
            PauliString = np.kron(PauliString, lista[temp[i][j]])
        lol = PauliString @ matrix
        value = np.trace(lol) ** 4
        summ += value
    final = (1.0 / ((2.0)**N)) * summ
    return final.real


def process_tableaux_chunk(chunk, FT, matrix_zero, n):
    unique_counts = {}
    magic = 0.0
    for idx, tab in enumerate(chunk):
        numerical_Clifford = tab.to_unitary_matrix(endian="little")
        matrix = FT @ numerical_Clifford @ matrix_zero @ np.conjugate(numerical_Clifford.T) @ np.conjugate(FT.T)
        value_magic = compute_magic_numerically_from_unitary_num(matrix, n)
        value_magic = np.round(value_magic, 5)
        unique_counts[value_magic] = unique_counts.get(value_magic, 0) + 1
        magic += -np.log2(value_magic)
    return unique_counts, magic


def merge_counts(dicts):
    merged = {}
    for d in dicts:
        for k, v in d.items():
            merged[k] = merged.get(k, 0) + v
    return merged


def chunkify(lst, n):
    """Split list into n roughly equal chunks"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == "__main__":
    n = 3
    FT = FourierMatrix(n)
    matrix_zero = np.zeros([int(2**n), int(2**n)], dtype=complex)
    matrix_zero[0, 0] = 1

    total_tabs = list(stim.Tableau.iter_all(n))
    num_chunks = min(os.cpu_count(), 6)  # Cap at 6 workers or less
    chunks = chunkify(total_tabs, num_chunks)

    print(f"Using {num_chunks} processes to handle {len(total_tabs)} tableaux.")

    results = []
    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        futures = [executor.submit(process_tableaux_chunk, chunk, FT, matrix_zero, n) for chunk in chunks]
        for f in tqdm(futures):
            results.append(f.result())

    all_counts = merge_counts([res[0] for res in results])
    total_magic = sum([res[1] for res in results])
    totCliffordUnitaries = cardinality_Cn_bar(n)

    print("--" * 30)
    counts_list = [[cnt, val] for val, cnt in all_counts.items()]
    print("Unique counts of value_magic:", counts_list)
    print("Average magic:", total_magic / totCliffordUnitaries)
