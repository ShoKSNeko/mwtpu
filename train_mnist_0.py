""" 単一ワーカでのトレーニング """
from sys import argv
from os import environ
import libtpu
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ['NEXT_PLUGGABLE_DEVICE_USE_C_API'] = 'true'
environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = libtpu.get_library_path()

import tensorflow as tf

import twolayer_model_0 as twolayer

def mnist_train_dataset():
    (x_train, y_train), test = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train / 255., y_train))
    dataset = dataset.shuffle(dataset.cardinality())
    dataset = dataset.batch(32, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train_mnist(strategy, savepath):
    dataset = mnist_train_dataset()
    model = twolayer.compiled_model(strategy)
    model.fit(dataset, epochs=10)
    model.save(savepath)

def tpu_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('local')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(topology, num_replicas=topology.num_tpus_per_task)
    strategy = tf.distribute.TPUStrategy(resolver, experimental_device_assignment=device_assignment)
    return strategy

if __name__ == "__main__":
    train_mnist(tpu_strategy(), *argv[1:])
