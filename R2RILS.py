"""
WRITTEN BY BAUCH & NADLER / 2020
"""


import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from sklearn.preprocessing import normalize


def R2RILS(R, omega, rank, max_iter=50, early_stopping_tol=1e-15, smart_tol=True):
    """
    :param R: Input matrix. Unobserved entries should be zero
    :type R: np.ndarray
    :param omega: Mask matrix. 1 on observed entries, 0 on unobserved.
    :type omega: np.ndarray
    :param rank: Underlying rank matrix.
    :param max_iter: Maximal number of iterations to perform.
    :type max_iter: int
    :param early_stopping_tol: tolerance to use for early stopping.
    :type early_stopping_tol: float
    :param smart_tol: Restrict tolerance for
    :return: R2RILS's estimate.
    """
    # initial estimate
    (U, _, V) = linalg.svds(R, k=rank, tol=1e-16)
    U = U.T
    n_1 = np.shape(U)[1]
    n_2 = np.shape(V)[1]
    num_visible_entries = np.count_nonzero(omega)
    iter_num = 0
    # generate sparse indices to accelerate future operations.
    sparse_matrix_rows, sparse_matrix_columns = generate_sparse_matrix_entries(omega, rank, n_1, n_2)
    A = generate_sparse_A(U, V, omega, sparse_matrix_rows, sparse_matrix_columns, num_visible_entries, n_1, n_2, rank)
    A, scale_vec = rescale_A(A)
    b = generate_b(R, omega, n_1, n_2)
    # find the least norm solution to Ax = b.
    x = linalg.lsqr(A, b)[0]
    x = x * scale_vec
    objective = 1
    while iter_num < max_iter and objective > early_stopping_tol:
        iter_num += 1
        x = convert_x_representation(x, rank, n_1, n_2)
        objective = np.sqrt(
            np.linalg.norm((get_estimated_value(x, U, V, rank, n_1, n_2) - R) * omega, ord='fro') ** 2 * (
                    1. / num_visible_entries))
        U_tilde, V_tilde = get_U_V_from_solution(x, rank, n_1, n_2)
        # ColNorm U_tilde, V_tilde
        U_tilde = normalize(U_tilde, axis=1)
        V_tilde = normalize(V_tilde, axis=1)

        U = normalize(U + U_tilde, axis=1)
        V = normalize(V + V_tilde, axis=1)

        A = generate_sparse_A(U, V, omega, sparse_matrix_rows, sparse_matrix_columns, num_visible_entries, n_1, n_2,
                              rank)
        A, scale_vec = rescale_A(A)
        tol = 1e-16
        if smart_tol:
            tol = min(objective ** 2, 1e-5)
        x = linalg.lsqr(A, b, atol=tol, btol=tol)[0]
        x = x * scale_vec

    x = convert_x_representation(x, rank, n_1, n_2)
    estimate = get_estimated_value(x, U, V, rank, n_1, n_2)
    return estimate


def rescale_A(A):
    A = sparse.csc_matrix(A)
    scale_vec = 1. / linalg.norm(A, axis=0)
    return normalize(A, axis=0), scale_vec


def convert_x_representation(x, rank, n_1, n_2):
    recovered_x = np.array([x[k * rank + i] for i in range(rank) for k in range(n_2)])
    recovered_y = np.array([x[rank * n_2 + j * rank + i] for i in range(rank) for j in range(n_1)])
    return np.append(recovered_x, recovered_y)


def generate_sparse_matrix_entries(omega, rank, n_1, n_2):
    row_entries = []
    columns_entries = []
    row = 0
    for j in range(n_1):
        for k in range(n_2):
            if 0 != omega[j][k]:
                # add indices for U entries
                for l in range(rank):
                    columns_entries.append(k * rank + l)
                    row_entries.append(row)
                # add indices for V entries
                for l in range(rank):
                    columns_entries.append((n_2 + j) * rank + l)
                    row_entries.append(row)
                row += 1
    return row_entries, columns_entries


def generate_sparse_A(U, V, omega, row_entries, columns_entries, num_visible_entries, n_1, n_2, rank):
    U_matrix = np.array(U).T
    V_matrix = np.array(V).T
    # we're generating row by row
    data_vector = np.concatenate(
        [np.concatenate([U_matrix[j], V_matrix[k]]) for j in range(n_1) for k in range(n_2) if 0 != omega[j][k]])
    return sparse.csr_matrix(sparse.coo_matrix((data_vector, (row_entries, columns_entries)),
                                               shape=(num_visible_entries, rank * (n_1 + n_2))))


def generate_b(R, omega, n_1, n_2):
    return np.array([R[j][k] for j in range(n_1)
                     for k in range(n_2) if 0 != omega[j][k]])


def get_U_V_from_solution(x, rank, n_1, n_2):
    V = np.array([x[i * n_2:(i + 1) * n_2] for i in range(rank)])
    U = np.array([x[rank * n_2 + i * n_1: rank * n_2 + (i + 1) * n_1] for i in range(rank)])
    return U, V


def get_estimated_value(x, U, V, rank, n_1, n_2):
    # calculate U's contribution
    estimate = np.sum(
        [np.dot(U[i].reshape(n_1, 1), np.array(x[i * n_2:(i + 1) * n_2]).reshape(1, n_2)) for i in
         range(rank)],
        axis=0)
    # calculate V's contribution
    estimate += np.sum(
        [np.dot(x[rank * n_2 + i * n_1: rank * n_2 + (i + 1) * n_1].reshape(n_1, 1),
                V[i].reshape(1, n_2))
         for i in range(rank)], axis=0)
    return estimate
