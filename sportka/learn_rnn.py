import csv
from datetime import date, datetime

import ephem
import numpy as np
from sportka.download import download_data_from_sazka
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class draw_history():

    draws = []

    def __init__(self):
        download_data_from_sazka()
        with open('/tmp/sportka.csv', newline='') as csvfile:
            the_reader = csv.reader(csvfile, delimiter=';')
            is_header = True
            for row in the_reader:
                if not is_header:
                    self.draws.append(draw(row, draw_history=self))

                is_header = False

class draw():

    def __init__(self, row, draw_history):
        try:
            print(row)
            self.date = datetime.strptime(row[0], '%d. %m. %Y').date()
            self.week = int(row[2])
            self.week_day = int(row[3])
            self.first = [int(x) for x in row[4:11]]
            self.second = [int(x) for x in row[11:18]]
            self.draw_history = draw_history
            #self.noise = NOISE

            print('>OK')
        except:
            print('>ERROR')

    @property
    def x_train(self):
        return date_to_x(self.date)

    @property
    def y_train_1(self):
        probability_first = np.array([1.0 if number in self.first else 0 for number in range(1, 50)])
        return probability_first / 7.0

    @property
    def y_train_2(self):
        probability_second = np.array([1.0 if number in self.second else 0 for number in range(1, 50)])
        return probability_second / 7.0

    @property
    def y_train_pairs_1(self):
        probability_first = np.array([
            0 if i == j else 1 if (i in self.first and j in self.first) else 0
            for i in range(1, 50)
            for j in range(1, 50)
        ])
        return probability_first / 49.0

    @property
    def y_train_pairs_2(self):
        probability_second = np.array([
            0 if i == j else 1 if (i in self.second and j in self.second) else 0
            for i in range(1, 50)
            for j in range(1, 50)
        ])
        return probability_second / 49.0

    @property
    def x_train_history_1(self):
        difference = 1
        index = self.draw_history.draws.index(self)
        history_index = index - difference

        if history_index >= 0:
            return self.draw_history.draws[history_index].y_train_1 #+ next(self.noise)
        else:
            return np.array([1/49.0 for number in range(1, 50)])

    @property
    def x_train_history_2(self):
        difference = 1
        index = self.draw_history.draws.index(self)
        history_index = index - difference
        if history_index >= 0:
            return self.draw_history.draws[history_index].y_train_2 #+ next(self.noise)
        else:
            return np.array([1/49.0 for number in range(1, 50)])



def date_to_x(date):

    #consider also some astrological data
    previous_new_moon = ephem.previous_new_moon(date)
    next_new_moon = ephem.next_new_moon(date)
    relative_lunation = (ephem.Date(date) - previous_new_moon) / (next_new_moon - previous_new_moon)

    return np.array([date.day / 31.0, date.month / 12.0, date.year / 2019.0, date.weekday() / 6.0, relative_lunation])


def learn_and_predict_keras(all_batches, iterations=20):
    batch_size = 1
    size_train=len(all_batches) - 1
    test_size = 10
    n_input = 49
    n_output = 49


    x_train = np.array(all_batches[:-2])
    y_train = np.array(all_batches[2:])

    validation_indexes = [random.choice(range(size_train)) for _ in range(10)]
    x_test = np.array([x_train[i] for i in validation_indexes])
    y_test = np.array([y_train[i] for i in validation_indexes])
    test_data = None

    model = keras.Sequential()
    #model.add(layers.Input(shape=(None, ), input_dim=n_input))
    model.add(layers.Embedding(input_dim=n_input, output_dim=20))

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(layers.GRU(256, return_sequences=True))

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(layers.SimpleRNN(n_output))

    #model.add(layers.Dense(units=n_output))

    model.summary()

    #model = build_model(allow_cudnn_kernel=True)

    model.compile(
        loss='categorical_crossentropy', #keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="sgd",
        metrics=["accuracy"],
    )

    model.fit(
        x_train, y_train, validation_data=test_data, batch_size=batch_size, epochs=1
    )



    return model

########################################################################################################################
############################## main program ############################################################################
########################################################################################################################
DATE_PREDICT = '10.03.2021'

dh = draw_history()
print(dh)

all_batches = [getattr(draw, attribute) for draw in dh.draws for attribute in ['x_train_history_1', 'x_train_history_2']]


rnn_model = learn_and_predict_keras(all_batches)
predicted = rnn_model.predict(all_batches[-1])

print(predicted)