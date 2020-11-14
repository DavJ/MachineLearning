"""
do not use probabilities but use directly numbers

"""
import csv
from datetime import date, datetime
import tensorflow as tf
import ephem

import numpy as np
import math
import random


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
        return np.array(self.first)

    @property
    def y_train_2(self):
        return np.array(self.second)

    @property
    def y_train_random(self):
        random_numbers = []
        for _  in range(7):
            while True:
                rn = math.ceil(49*random.random())
                if rn not in random_numbers:
                    random_numbers.append(rn)
                    break
                else:
                    continue
        return np.array(random_numbers)

    @property
    def x_train_history_1(self):
        difference = 1
        index = self.draw_history.draws.index(self)
        history_index = index - difference
        if history_index >= 0:
            return self.draw_history.draws[history_index].y_train_1
        else:
            return self.y_train_random

    @property
    def x_train_history_2(self):
        difference = 1
        index = self.draw_history.draws.index(self)
        history_index = index - difference
        if history_index >= 0:
            return self.draw_history.draws[history_index].y_train_2
        else:
            return self.y_train_random

    @property
    def observer(self):
        return sazka_building


def date_to_x(date):

    #consider also some astrological data
    previous_new_moon = ephem.previous_new_moon(date)
    next_new_moon = ephem.next_new_moon(date)
    relative_lunation = (ephem.Date(date) - previous_new_moon) / (next_new_moon - previous_new_moon)

    return np.array([date.day / 31.0, date.month / 12.0, date.year / 2019.0, date.weekday() / 6.0, relative_lunation])

def normalize(array):
    return tf.map_fn(lambda x: x % 49 + 1, array)

def my_loss(x_true, y_pred):
    # y_norm = normalize(y_pred)
    y_norm = y_pred
    x_duplicite = 0
    y_duplicite = 0
    #y_duplicite = len(y_norm) - len(numpy.unique(y_norm))
    duplicities = x_duplicite + y_duplicite
    matcher = np.in1d(x_true, y_norm)
    matching = sum([1 for m in matcher if m])

    f = math.factorial
    n = 6
    loss = math.pow(f(n + duplicities) / f(matching) / f(n + duplicities-matching) + random.random() - 0.5, 2)

    return tf.convert_to_tensor(loss, dtype=tf.float32)


    #r eturn tf.keras.backend.mean(loss, axis=None, keepdims=False)


def learn_and_predict_sportka(x_train, y_train, x_predict, depth=256, epochs=15):

    inputs = tf.keras.Input(shape=(19,))  # Returns a placeholder tensor

    x = tf.keras.layers.Dense(128, activation='sigmoid',
                              kernel_initializer='random_normal',
                              bias_initializer='random_normal')(inputs)

    for i in range(1, depth - 2):
        x = tf.keras.layers.Dense(128, activation='sigmoid',
                                  kernel_initializer='random_normal',
                                  bias_initializer='random_normal'
                                  )(x)
        x = tf.keras.layers.Dropout(0.4)(x)

    for i in range(1):
        x = tf.keras.layers.Dense(6, activation='sigmoid',
                                  kernel_initializer='random_normal',
                                  bias_initializer='random_normal'
                                  )(x)
        x = tf.keras.layers.Dropout(0.4)(x)

    predictions = tf.keras.layers.Dense(14, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    # The compile step specifies the training configuration.
    model.compile(optimizer='adam', loss='mse')

    model.fit(x=x_train, y=y_train, epochs=epochs)
    return model.predict(x_predict)

########################################################################################################################
############################## main program ############################################################################
########################################################################################################################

DATE_PREDICT = '16.11.2020'

dh = draw_history()
print(dh)

x_predict = np.array([date_to_x(datetime.strptime(DATE_PREDICT, '%d.%m.%Y').date())])
x_predict_draw_1 = np.array([dh.draws[-1].y_train_1])
x_predict_draw_2 = np.array([dh.draws[-1].y_train_2])
x_predict_all = [np.concatenate((x_predict, x_predict_draw_1, x_predict_draw_2), axis=1)]

x_train_all = np.array(
    [np.concatenate((draw.x_train, draw.x_train_history_1, draw.x_train_history_2), axis=0) for draw in dh.draws])

y_train_1 = np.array([draw.y_train_1 for draw in dh.draws])
y_train_2 = np.array([draw.y_train_2 for draw in dh.draws])
y_train_all = [np.concatenate((y_train_1, y_train_2), axis=1)]

y_predict = learn_and_predict_sportka(x_train_all, y_train_all, x_predict_all, depth=128, epochs=500)

print('both draws ')
print('14 best numbers for {}\n: {}\n\n'.format(DATE_PREDICT, y_predict))

