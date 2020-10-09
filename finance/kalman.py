from pykalman import KalmanFilter
import numpy as np
from finance.alphavantage.sync import read_json
from datetime import datetime, timedelta

NSAMPLES = 120
NDIM = 30
SYMBOL = 'AAPL'
DAYS_BACK_HISTORY = 25


def state_transition_matrix(n_dim_y, n_dim_x):
    if n_dim_y == 1:
        return np.array([[1.0 / np.math.factorial(index) for index in range(0, n_dim_x)]])

    else:
        matrix = state_transition_matrix(n_dim_y-1, n_dim_x)
        return np.vstack([matrix, np.hstack([np.array([0]), matrix[-1][:-1]])])

def read_data(samples=10, offset=0, derivations=3, symbol='AAPL', field='4. close'):

    def append_derivatives(matrix, remaining_derivatives):
        if remaining_derivatives > 0:
            new_matrix = np.vstack([matrix, np.gradient(matrix[-1])])
            return append_derivatives(new_matrix, remaining_derivatives-1)
        else:
            return matrix

    response = read_json(symbol)

    sorted_keys = reversed(list(response.keys())[offset : samples + offset]) #TODO implement sorting, don't rely that data are always sorted

    all_data = {key: response[key][field] for key in response.keys()}
    data_1d = np.array([[all_data[key] for key in sorted_keys]], dtype=float)
    print(f'latest date in downloaded data for symbol {symbol}: {list(response.keys())[0]}')
    last_date=datetime.fromisoformat(list(response.keys())[offset])
    return last_date, all_data, append_derivatives(data_1d, derivations).transpose()

def predict_means(last_state, kalman_filtr, number_of_days=1):

    def displacement_power_vector(n_dim, number_of_days):
        displacement_vector = number_of_days * np.ones(n_dim)
        powers = np.array(range(n_dim))
        return np.power(displacement_vector, powers).transpose()

    n_dim = len(last_state)
    tp_generating_matrix = state_transition_matrix(n_dim, n_dim)
    partial_differences = np.multiply(last_state, displacement_power_vector(n_dim, number_of_days))
    predicted_state = tp_generating_matrix.dot(partial_differences)

    return predicted_state


last_date, all_prices, measurements = read_data(samples=NSAMPLES, offset=DAYS_BACK_HISTORY, derivations=NDIM-1, symbol=SYMBOL)

kf_smoothing = KalmanFilter(n_dim_obs=NDIM, n_dim_state=NDIM)
results_smoothed = kf_smoothing.em(measurements, n_iter=20).smooth(measurements)[0]

F = state_transition_matrix(NDIM, NDIM)
kf_designed = KalmanFilter(n_dim_obs=NDIM,
                           n_dim_state=NDIM,
                           transition_matrices=F).em(measurements, n_iter=20)

(filtered_state_means, filtered_state_covariances) = kf_designed.filter(measurements)

#print(filtered_state_means[-1])
#print(filtered_state_covariances[-1])
print('*************************************** OBSERVATIONS **********************************************************')
last_state1 = filtered_state_means[-1]
last_state2 = results_smoothed[-1]

print(f'Last observed day: {str(last_date)}')
print(f'Last observed price: {measurements[-1][0]}')
print(f'Last mean price by filter1: {last_state1[0]}')
print(f'Last mean price by filter2: {last_state2[0]}')

print('\n')
print(f'Days history offset {DAYS_BACK_HISTORY}:')
print('\n')
print('*************************************** PREDICTIONS ***********************************************************')
for day_delta in range(10):
    predicted_price_designed = predict_means(last_state=last_state1,
                                    kalman_filtr=kf_designed,
                                    number_of_days=day_delta)[0]

    predicted_price_smoothing = predict_means(last_state=last_state2,
                                    kalman_filtr=kf_smoothing,
                                    number_of_days=day_delta)[0]
    prediction_date = last_date + timedelta(day_delta)
    real_price = all_prices[str(prediction_date.date())] if str(prediction_date.date()) in all_prices.keys() else None

    print(f'day {str(prediction_date)} prediction for {SYMBOL}: {predicted_price_designed:.2f},'
          f' {predicted_price_smoothing:.2f}, {0.5*(predicted_price_designed + predicted_price_smoothing):.2f}'
          f'---> {str(real_price)}')

print('****************************************** END ****************************************************************')

