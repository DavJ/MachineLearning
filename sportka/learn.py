import csv
from datetime import date, datetime
import tensorflow as tf
import ephem

import numpy as np


class sazka_building(ephem.Observer):
    def __init__(self, date):
        super.__init__()
        self.lon = '14.4963524'
        self.lat = '50.0986794'
        self.elevation = 234
        self.date = date


class draw_history(object):

    draws = []

    def __init__(self):
        with open('sportka.csv', newline='') as csvfile:
            the_reader = csv.reader(csvfile, delimiter=';')
            is_header = True
            for row in the_reader:
                if not is_header:
                    self.draws.append(draw(row, draw_history=self))

                is_header = False


class draw(object):
    def __init__(self, row, draw_history):
        try:
            print(row)
            self.date = datetime.strptime(row[0], '%d. %m. %Y').date()
            self.week = int(row[2])
            self.week_day = int(row[3])
            self.first = [int(x) for x in row[4:11]]
            self.second = [int(x) for x in row[11:18]]
            self.draw_history = draw_history
            #self.observer = sazka_building(self.date)

            print('>OK')
        except:
            print('>ERROR')

    @property
    def x_train(self):
        return date_to_x(self.date)

    @property
    def y_train_1(self):
        probability_first = np.array(
            [1.0 if number in self.first else 0 for number in range(1, 50)])
        return probability_first

    @property
    def y_train_2(self):
        probability_second = np.array(
            [1.0 if number in self.second else 0 for number in range(1, 50)])
        return probability_second

    @property
    def y_train_pairs_1(self):
        probability_first = np.array([
            0 if i == j else 1 if (i in self.first and j in self.first) else 0
            for i in range(1, 50) for j in range(1, 50)
        ])
        return probability_first

    @property
    def y_train_pairs_2(self):
        probability_second = np.array([
            0 if i == j else 1 if
            (i in self.second and j in self.second) else 0
            for i in range(1, 50) for j in range(1, 50)
        ])
        return probability_second

    @property
    def x_train_history_1(self):
        difference = 1
        index = self.draw_history.draws.index(self)
        history_index = index - difference
        if history_index >= 0:
            return self.draw_history.draws[history_index].y_train_1
        else:
            return np.array([0 for number in range(1, 50)])

    @property
    def x_train_history_2(self):
        difference = 1
        index = self.draw_history.draws.index(self)
        history_index = index - difference
        if history_index >= 0:
            return self.draw_history.draws[history_index].y_train_2
        else:
            return np.array([0 for number in range(1, 50)])

    @property
    def observer(self):
        return sazka_building


def date_to_x(date):

    #consider also some astrological data
    previous_new_moon = ephem.previous_new_moon(date)
    next_new_moon = ephem.next_new_moon(date)
    relative_lunation = (ephem.Date(date) - previous_new_moon) / (
        next_new_moon - previous_new_moon)

    return np.array([
        date.day / 31.0, date.month / 12.0, date.year / 2019.0,
        date.weekday() / 6.0, relative_lunation
    ])


def learn_and_predict_sportka(x_train,
                              y_train_both,
                              x_predict,
                              depth=1,
                              depth_wide=2,
                              epochs=15):

    inputs = tf.keras.Input(shape=(103, ))  # Returns a placeholder tensor

    x = tf.keras.layers.Dense(128, activation='relu')(inputs)

    for i in range(1, depth - depth_wide):
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)

    for i in range(1, depth_wide):
        x = tf.keras.layers.Dense(2450, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)

    predictions = tf.keras.layers.Dense(2450, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.AdamOptimizer(0.0005),
                  loss='mse',
                  metrics=['msle', 'mean_squared_error'])

    model.fit(x=x_train, y=y_train_both, epochs=epochs)
    return model.predict(x_predict)


def best_numbers(y_predict, n=6):
    numbers_vs_chances = ((i + 1, y_predict[0][i]) for i in range(49))
    sorted_numbers = sorted(numbers_vs_chances,
                            key=lambda x: x[1],
                            reverse=True)
    return [key for key in sorted_numbers[0:n]]


def best_pairs(y_predict_pairs, n=30):
    pairs_vs_chances = [(i + 1, j + 1, y_predict_pairs[0][i + j * 49])
                        for i in range(49) for j in range(49)]
    sorted_pairs = sorted(pairs_vs_chances, key=lambda x: x[2], reverse=True)
    return [key for key in sorted_pairs[0:n]]


########################################################################################################################
############################## main program ############################################################################
########################################################################################################################

DATE_PREDICT = '23.6.2019'

dh = draw_history()
print(dh)

x_predict = np.array(
    [date_to_x(datetime.strptime(DATE_PREDICT, '%d.%m.%Y').date())])
x_predict_draw_1 = np.array([dh.draws[-1].y_train_1])
x_predict_draw_2 = np.array([dh.draws[-1].y_train_2])
x_predict_all = [
    np.concatenate((x_predict, x_predict_draw_1, x_predict_draw_2), axis=1)
]

x_train_all = np.array([
    np.concatenate(
        (draw.x_train, draw.x_train_history_1, draw.x_train_history_2), axis=0)
    for draw in dh.draws
])

y_train_1 = np.array([draw.y_train_1 for draw in dh.draws])
y_train_2 = np.array([draw.y_train_2 for draw in dh.draws])
y_train_pairs_1 = np.array([draw.y_train_pairs_1 for draw in dh.draws])
y_train_pairs_2 = np.array([draw.y_train_pairs_1 for draw in dh.draws])
y_train_all_1 = np.array([
    np.concatenate((draw.y_train_1, draw.y_train_pairs_1), axis=0)
    for draw in dh.draws
])
y_train_all_2 = np.array([
    np.concatenate((draw.y_train_2, draw.y_train_pairs_2), axis=0)
    for draw in dh.draws
])

y_predict_all_1 = learn_and_predict_sportka(x_train_all,
                                            y_train_all_1,
                                            x_predict_all,
                                            depth=128,
                                            epochs=10)
y_predict_numbers_1 = y_predict_all_1[:49]

y_predict_all_2 = learn_and_predict_sportka(x_train_all,
                                            y_train_all_2,
                                            x_predict_all,
                                            depth=128,
                                            epochs=10)
y_predict_numbers_2 = y_predict_all_2[:49]

print('first draw ')
print('best numbers for {}\n: {}\n\n'.format(
    DATE_PREDICT, best_numbers(y_predict_numbers_1, 6)))
print('all numbers\n: {}\n\n'.format(best_numbers(y_predict_numbers_1, 49)))

print('second draw {}')
print('best numbers for {}\n: {}\n\n'.format(
    DATE_PREDICT, best_numbers(y_predict_numbers_2, 6)))
print('all numbers\n: {}\n\n'.format(best_numbers(y_predict_numbers_2, 49)))

print('combined :')
print('best numbers for {}\n: {}\n\n'.format(
    DATE_PREDICT, best_numbers(y_predict_numbers_1 + y_predict_numbers_2, 6)))
print('all numbers\n: {}\n\n'.format(
    best_numbers(y_predict_numbers_1 + y_predict_numbers_2, 49)))
