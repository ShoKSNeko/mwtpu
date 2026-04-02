""" 各ワーカでばらばらにトレーニング """
from sys import argv
from os import environ
import libtpu
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ['NEXT_PLUGGABLE_DEVICE_USE_C_API'] = 'true'
environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = libtpu.get_library_path()

import tensorflow as tf

from mw_util import my_worker_id, num_workers
import twolayer_model_0 as twolayer

def mnist_train_dataset(workernum, worker_id):
    (x_train, y_train), test = tf.keras.datasets.mnist.load_data()
    myndx = y_train % workernum == worker_id
    dataset = tf.data.Dataset.from_tensor_slices((x_train[myndx] / 255., y_train[myndx]))
    dataset = dataset.shuffle(dataset.cardinality())
    dataset = dataset.batch(32, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train_mnist(strategy, saveprefix):
    workernum, worker_id = num_workers(), my_worker_id()
    dataset = mnist_train_dataset(workernum, worker_id)
    model = twolayer.compiled_model(strategy)
    model.fit(dataset, epochs=10, verbose=2)
    model.save_weights(f"{saveprefix}_{worker_id}.weights.h5")

def tpu_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('local')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(topology, num_replicas=topology.num_tpus_per_task)
    strategy = tf.distribute.TPUStrategy(resolver, experimental_device_assignment=device_assignment)
    return strategy

if __name__ == "__main__":
    train_mnist(tpu_strategy(), *argv[1:])
