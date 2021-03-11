import csv
from datetime import date, datetime

from sportka.download import download_data_from_sazka
import random
import numpy as np
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
            return 7.0*self.draw_history.draws[history_index].y_train_1
        else:
            return np.array([0 for number in range(1, 50)])

    @property
    def x_train_history_2(self):
        difference = 1
        index = self.draw_history.draws.index(self)
        history_index = index - difference
        if history_index >= 0:
            return 7.0*self.draw_history.draws[history_index].y_train_2
        else:
            return np.array([0 for number in range(1, 50)])

def learn_and_predict_keras(all_batches, iterations=20):
    batch_size = None
    size_train = len(all_batches) - 1
    test_size = 10
    n_input = 98
    n_output = 98

    x_train = np.array(all_batches[:-1])
    y_train = np.array(all_batches[1:])

    validation_indexes = [random.choice(range(size_train)) for _ in range(10)]
    x_test = np.array([x_train[i] for i in validation_indexes])
    y_test = np.array([y_train[i] for i in validation_indexes])
    test_data = (x_test, y_test)
    #test_data = None

    model = keras.Sequential()

    model.add(layers.Embedding(input_dim=n_input, output_dim=8))

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(layers.GRU(256, return_sequences=True))

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 96)
    model.add(layers.SimpleRNN(n_output))

    model.add(layers.Dense(n_output))

    model.summary()

    #model = build_model(allow_cudnn_kernel=True)

    model.compile(
        loss='categorical_crossentropy',                  #keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="sgd",
        metrics=["accuracy"],
    )

    model.fit(
        x_train, y_train, validation_data=test_data, batch_size=batch_size, epochs=1
    )

    return model


def numbers_vs_chances(y_predict, n=49):
    norm = sum([y_predict[0][i] for i in range(98)])
    numbers_vs_chances = ((i + 1, 0.5*(y_predict[0][i] + y_predict[0][i+49])/norm) for i in range(49))
    sorted_numbers = sorted(numbers_vs_chances, key=lambda x: x[1], reverse=True)
    return [key for key in sorted_numbers[0:n]]

def best_numbers(numbers_vs_chances):
    return [n[0] for n in sorted(numbers_vs_chances, key=lambda x: x[1], reverse=True)[:6]]

def random_predict(numbers_vs_chances):
    weights = [w[1] for w in sorted(numbers_vs_chances, key=lambda x: x[0], reverse=False)]
    proper_weights = (weights - min(weights)) / max(weights - min(weights))
    while True:
      choice=random.choices(range(1, 50), weights=proper_weights, k=6)
      if max([choice.count(c) for c in choice]) == 1:
        break
    return sorted(choice)

########################################################################################################################
############################## main program ############################################################################
########################################################################################################################
DATE_PREDICT = '10.03.2021'

dh = draw_history()
print(dh)

all_batches = [np.concatenate((draw.x_train_history_1, draw.x_train_history_2)) for draw in dh.draws]

rnn_model = learn_and_predict_keras(all_batches)
predicted = rnn_model.predict(all_batches[-1])

print(predicted)
numbers_by_chances = numbers_vs_chances(y_predict=predicted)


print(numbers_by_chances)

print('\n\n\nRECOMMENDED NUMBERS')
for sloupec in range(1, 13):
    if sloupec == 1:
       recommend = best_numbers(numbers_by_chances)
    else:
       recommend = random_predict(numbers_by_chances)

    print(f'{sloupec}. - {recommend}')






