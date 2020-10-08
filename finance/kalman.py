from pykalman import KalmanFilter
import numpy as np
from finance.alphavantage.sync import read_json

NSAMPLES = 100
NDIM = 20
SYMBOL = 'AAPL'


def state_transition_matrix(n_dim_y, n_dim_x):
    if n_dim_y == 1:
        return np.array([[1.0 / np.math.factorial(index) for index in range(0, n_dim_x)]])

    else:
        matrix = state_transition_matrix(n_dim_y-1, n_dim_x)
        return np.vstack([matrix, np.hstack([np.array([0]), matrix[-1][:-1]])])


def read_data(samples=10, derivations=3, symbol='AAPL', field='4. close'):

    def append_derivatives(matrix, remaining_derivatives):
        if remaining_derivatives > 0:
            new_matrix = np.vstack([matrix, np.gradient(matrix[-1])])
            return append_derivatives(new_matrix, remaining_derivatives-1)
        else:
            return matrix

    response = read_json(symbol)

    sorted_keys = reversed(list(response.keys())[:samples]) #TODO implement sorting, don't rely that data are always sorted
    data_1d = np.array([[response[key][field] for key in sorted_keys]], dtype=float)
    print(f'latest date: {list(response.keys())[0]}')
    return append_derivatives(data_1d, derivations).transpose()

measurements = read_data(samples=NSAMPLES, derivations=NDIM-1, symbol=SYMBOL)

kf_smoothing = KalmanFilter(n_dim_obs=NDIM, n_dim_state=NDIM)
results_smoothed = kf_smoothing.em(measurements, n_iter=20).smooth(measurements)[0]

F = state_transition_matrix(NDIM, NDIM)
kf_designed = KalmanFilter(n_dim_obs=NDIM,
                           n_dim_state=NDIM,
                           transition_matrices=F).em(measurements, n_iter=20)
(filtered_state_means, filtered_state_covariances) = kf_designed.filter(measurements)


print(filtered_state_means[-1])
print(filtered_state_covariances[-1])

print(f'one day prediction for {SYMBOL}: {sum(filtered_state_means[-1])}')
