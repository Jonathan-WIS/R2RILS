import numpy as np
from scipy.stats import ortho_group
from R2RILS import R2RILS


def run_demo():
    n_1 = 200
    n_2 = 300
    rank = 5
    oversampling = 3.
    p = oversampling * rank * (n_1 + n_2 - rank) / (n_1 * n_2)
    singular_valeus = [1] * rank
    for i in range(20):
        U, V, omega, _ = generate_experiment_data(n_1, n_2, singular_valeus, p)
        X0 = np.dot(U, V.T)
        X = X0 * omega
        X_hat = R2RILS(X, omega, rank)
        num_visible_entries = np.count_nonzero(omega)
        unobserved_RMSE = np.sqrt(
            np.linalg.norm((X_hat - X0) * (1 - omega), ord='fro') ** 2 * (1. / (n_1 * n_2 - num_visible_entries)))
        observed_RMSE = np.sqrt(
            np.linalg.norm((X_hat - X0) * omega, ord='fro') ** 2 * (1. / num_visible_entries))
        print('iteration: {}, unobserved RMSE: {}, observed RMSE: {}'.format(i, unobserved_RMSE, observed_RMSE))


def generate_experiment_data(n_1, n_2, singular_values, p, noise_level=0):
    """
    Generate data for an experiment
    :param n_1: number of rows in matrix
    :param n_2: number of columns in matrix
    :param singular_values: singular values as an array
    :param p: probability of observing an entry
    :param noise_level: additive Gaussian noise standard deviation.
    :return: U, V, mask, noise such that
     - U x V' is a rank len(singular_values) matrix with non zero singular values equal to singular_values.
     - omega is the matrix of observed entries. With 1 indicating that an entry is observed and zero that it is not.
            omega is resampled until there are at least len(singular_values) visible entries in each column and row.
     - noise: an n_1xn_2 matrix with i.i.d Gaussian entries sampled with standard deviation noise_level.
    """
    rank = len(singular_values)
    U, V = generate_set_singular_values(n_1, n_2, rank, singular_values)
    # resample mask until there are enough measurements
    num_resamples = 0
    while True:
        num_resamples += 1
        omega = np.round((np.random.random((n_1, n_2)) + p) * 1. / 2)
        # count non zero on columns
        if min(np.count_nonzero(omega, axis=0)) < rank:
            print('resampling mask {}'.format(num_resamples))
            continue
        if min(np.count_nonzero(omega, axis=1)) < rank:
            print('resampling mask {}'.format(num_resamples))
            continue
        break
    noise = np.random.randn(n_1, n_2) * noise_level
    return U, V, omega, noise


def generate_set_singular_values(n_1, n_2, rank, singular_values):
    U = np.random.randn(n_1, rank)
    V = np.random.randn(n_2, rank)
    V = np.linalg.qr(V)[0]
    U = np.linalg.qr(U)[0]
    return U, singular_values * V


if __name__ == '__main__':
    run_demo()
