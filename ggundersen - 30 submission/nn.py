"""Predict eviction response variable.
"""

import data
from sklearn import metrics
from models.neural_network.neural_network import NeuralNetwork
import numpy as np
import sys


TYPE_TO_N_FEATURES = {
    'full'       : [12133, 6000],
    'pca'        : [56, 25],
    'handcrafted': [988, 450],
    'rlogistic'  : [15, 8]
}

DATASET_TYPE   = sys.argv[1]
FILENAME       = 'predict/predictions/predictions_%s_dataset_NN.csv' % DATASET_TYPE.upper()
NN_WEIGHTS_DIR = 'models/neural_network'

NUM_FEATURES   = TYPE_TO_N_FEATURES[DATASET_TYPE][0]
NUM_HIDDEN     = TYPE_TO_N_FEATURES[DATASET_TYPE][1]
NUM_OUTPUTS    = 2      # Evicted or not

LAYERS         = [NUM_FEATURES, NUM_HIDDEN, NUM_OUTPUTS]
RATE           = 3.0
EPOCHS         = 10
BATCH_SIZE     = 20

NUM_TESTS      = 1
# 1459 * 0.8 \approx 1167, round down so we can use batches.
NUM_SAMPLES    = 1160


def main():

    dataset = data.load_eviction(dataset_type=DATASET_TYPE)

    losses = []
    accs   = []
    precs  = []
    recs   = []

    with open(FILENAME, 'w+') as f:
        for i in range(NUM_TESTS):

            print('*' * 80)
            print('Round %s' % i)

            dataset.split()
            train(dataset)
            predictions = predict(dataset)
            loss, acc, prec, rec = evaluate(dataset.y_test, predictions)

            losses.append(loss)
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)

        loss_avg = np.array(losses).mean()
        accs_avg = np.array(accs).mean()
        precs_avg = np.array(precs).mean()
        recs_avg = np.array(recs).mean()

        print('Loss average: %s' % loss_avg)
        print('Accuracy average: %s' % accs_avg)
        print('Precision average: %s' % precs_avg)
        print('Recall average: %s' % recs_avg)

        output = '%s, %s, %s, %s, %s\n' % ('nn', loss_avg, accs_avg, precs_avg,
                                           recs_avg)
        f.write(output)


def train(dataset):
    X_train = dataset.X_train.as_matrix()[0:NUM_SAMPLES:]
    y_train_one_hots = dataset.y_train_one_hots[0:NUM_SAMPLES]
    print('Data loaded.')

    net = NeuralNetwork(LAYERS)
    net.train_SGD(X_train, y_train_one_hots, RATE, EPOCHS, BATCH_SIZE,
                 test_X=dataset.X_test.as_matrix(), test_y=dataset.y_test)

    print('Neural network trained.')
    np.savetxt('%s/weights1.txt' % NN_WEIGHTS_DIR, net.weights[0])
    np.savetxt('%s/weights2.txt' % NN_WEIGHTS_DIR, net.weights[1])
    print('Weights saved.')


def predict(dataset):
    net = NeuralNetwork(LAYERS)
    net.weights[0] = np.loadtxt('%s/weights1.txt' % NN_WEIGHTS_DIR)
    net.weights[1] = np.loadtxt('%s/weights2.txt' % NN_WEIGHTS_DIR)
    print('Data and weights loaded.')
    predictions = net.predict(dataset.X_test.as_matrix())
    return np.array([p.argmax() for p in predictions])


def evaluate(targs, preds):
    loss = metrics.mean_squared_error(targs, preds)
    acc  = metrics.accuracy_score(targs, preds)
    prec = metrics.precision_score(targs, preds)
    rec  = metrics.recall_score(targs, preds)
    print('Loss: %s' % loss)
    print('Accuracy: %s' % acc)
    print('Precision: %s' % prec)
    print('Recall: %s' % rec)
    return loss, acc, prec, rec


if __name__ == '__main__':
    main()
