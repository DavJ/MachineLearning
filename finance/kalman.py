from pykalman import KalmanFilter
import numpy as np
from finance.alphavantage.sync import read_json

NSAMPLES = 100
NDIM = 4

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
    return append_derivatives(data_1d, derivations).transpose()


measurements = read_data(samples=NSAMPLES, derivations=NDIM-1)

kf = KalmanFilter(n_dim_obs=NDIM, n_dim_state=NDIM)
results = kf.em(measurements, n_iter=20).smooth(measurements)[0]

#means, covariances = kf.filter(measurements)
pass