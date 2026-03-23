""" トレーニング結果の評価 """
from sys import argv
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf

import twolayer_model_0 as twolayer

def mnist_test_samples():
    train, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return [(int(yval), x_test[y_test == yval] / 255.) for yval in np.unique(y_test)]

def test_mnist(weightsprefix):
    samples = mnist_test_samples()
    model = twolayer.TwolayerModel()
    model(samples[0][1][:1])
    model.load_weights(f"{weightsprefix}.weights.h5")
    for y, x in samples:
        print(f"{y}: {np.count_nonzero(model.predict(x, None, 0).argmax(axis=1) == y) / x.shape[0]:f}")

if __name__ == "__main__":
    test_mnist(*argv[1:])
