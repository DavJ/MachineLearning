from pykalman import KalmanFilter
import numpy as np
#from alphavantage import data_json
from datetime import datetime
from config import NSAMPLES, NDIM, DAYS_BACK_HISTORY

def state_transition_matrix(n_dim_y, n_dim_x):
    if n_dim_y == 1:
        return np.array([[1.0 / np.math.factorial(index) for index in range(0, n_dim_x)]])

    else:
        matrix = state_transition_matrix(n_dim_y-1, n_dim_x)
        return np.vstack([matrix, np.hstack([np.array([0]), matrix[-1][:-1]])])

STATE_TRANSITION_MATRIX = state_transition_matrix(NDIM, NDIM)

def read_data(samples=10, offset=0, derivations=3, symbol='AAPL', field='4. close' , response=None):

    def append_derivatives(matrix, remaining_derivatives):
        if remaining_derivatives > 0:
            new_matrix = np.vstack([matrix, np.gradient(matrix[-1])])
            return append_derivatives(new_matrix, remaining_derivatives-1)
        else:
            return matrix
    time_series_daily = response['Time Series (Daily)']

    if offset == 0:
         sorted_keys = sorted([k for k in time_series_daily.keys()],
                         key=lambda x: datetime.fromisoformat(x))[-NSAMPLES::]
    else:
         sorted_keys = sorted([k for k in time_series_daily.keys()],
                             key=lambda x: datetime.fromisoformat(x))[-NSAMPLES-offset:-offset]

    all_data = {key: time_series_daily[key][field] for key in time_series_daily.keys()}
    data_1d = np.array([[all_data[key] for key in sorted_keys]], dtype=float)
    print(f'latest date in downloaded data for symbol {symbol}: {list(time_series_daily.keys())[0]}')
    last_date = datetime.fromisoformat(list(time_series_daily.keys())[offset])
    return last_date, all_data, append_derivatives(data_1d, derivations).transpose()

def predict_means(last_state, kalman_filtr, number_of_days=1):

    def displacement_power_vector(n_dim, number_of_days):
        displacement_vector = number_of_days * np.ones(n_dim)
        powers = np.array(range(n_dim))
        return np.power(displacement_vector, powers).transpose()

    n_dim = len(last_state)
    tp_generating_matrix = STATE_TRANSITION_MATRIX
    partial_differences = np.multiply(last_state, displacement_power_vector(n_dim, number_of_days))
    predicted_state = tp_generating_matrix.dot(partial_differences)

    return predicted_state

def predict_stock_price(symbol, days_delta, data_json):
    last_date, all_prices, measurements = read_data(samples=NSAMPLES, offset=0,
                                                    derivations=NDIM-1, symbol=symbol, response=data_json)
    kf_designed = KalmanFilter(n_dim_obs=NDIM,
                               n_dim_state=NDIM,
                               transition_matrices=STATE_TRANSITION_MATRIX).em(measurements, n_iter=20)

    (filtered_state_means, filtered_state_covariances) = kf_designed.filter(measurements)
    predicted_price = predict_means(last_state=filtered_state_means[-1],
                                    kalman_filtr=kf_designed,
                                    number_of_days=days_delta)[0]

    return predicted_price
