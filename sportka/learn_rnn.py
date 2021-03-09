import csv
from datetime import date, datetime
import tensorflow as tf
import ephem
import numpy as np
from sportka.download import download_data_from_sazka
import random

#REALIZATIONS = 10


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

    #@property
    #def noise(self, mean=0, deviation=0.5):
    #     yield np.random.normal(mean, deviation, 49)

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


def learn_and_predict_sportka(all_batches, iterations=20):
    n_windows = 2
    n_input = 49
    n_output = 49
    size_train = 201
    r_neuron=128

    X_batches = all_batches[::-1]
    y_batches = all_batches[1::]
    X_predict = all_batches[-1]

    ## 1. Construct the tensors
    X = tf.Variable(tf.float32, [None, n_windows, n_input])
    y = tf.Variable(tf.float32, [None, n_windows, n_output])

    ## 2. create the model
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=r_neuron, activation=tf.nn.relu)
    rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    stacked_rnn_output = tf.reshape(rnn_output, [-1, r_neuron])
    stacked_outputs = tf.layers.dense(stacked_rnn_output, n_output)
    outputs = tf.reshape(stacked_outputs, [-1, n_windows, n_output])

    ## 3. Loss + optimization
    learning_rate = 0.001

    loss = tf.reduce_sum(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    ## 4. Train
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        for iters in range(iterations):
           sess.run(training_op, feed_dict={X: X_batches, y: y_batches})
           if iters % 150 == 0:
               mse = loss.eval(feed_dict={X: X_batches, y: y_batches})
               print(iters, "\tMSE:", mse)

        y_pred = sess.run(outputs, feed_dict={X: X_predict})

    return y_pred


########################################################################################################################
############################## main program ############################################################################
########################################################################################################################
DATE_PREDICT = '10.03.2021'

dh = draw_history()
print(dh)
#REALIZATIONS = range(15)

all_batches = [[draw.x_train_history_1, draw.x_train_history_2] for draw in dh.draws]
y_batches = all_batches[1::]

y_predict = learn_and_predict_sportka(all_batches)


#x_predict = np.array([date_to_x(datetime.strptime(DATE_PREDICT, '%d.%m.%Y').date())])
#x_predict_draw_1 = np.array([dh.draws[-1].y_train_1])
#x_predict_draw_2 = np.array([dh.draws[-1].y_train_2])
#x_predict_all = [np.concatenate((x_predict, x_predict_draw_1, x_predict_draw_2), axis=1)]


#x_train_all = np.array([np.concatenate((draw.x_train, draw.x_train_history_1, draw.x_train_history_2), axis=0) for draw in dh.draws for realization in REALIZATIONS])

#y_train_1 = np.array([draw.y_train_1 for draw in dh.draws for realization in REALIZATIONS])
#y_train_2 = np.array([draw.y_train_2 for draw in dh.draws for realization in REALIZATIONS])

#y_predict_1 = learn_and_predict_sportka(x_train_all, y_train_1, x_predict_all, depth=128, epochs=150)
#y_predict_numbers_1 = y_predict_1[:49]
#y_predict_2 = learn_and_predict_sportka(x_train_all, y_train_2, x_predict_all, depth=128, epochs=150)
#y_predict_numbers_2 = y_predict_2[:49]
