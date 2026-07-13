""" 単一ワーカでのトレーニング """
""" カスタムトレーニングループ """
from sys import argv
from os import environ
import libtpu
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ['NEXT_PLUGGABLE_DEVICE_USE_C_API'] = 'true'
environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = libtpu.get_library_path()

import tensorflow as tf

import twolayer_model_1 as twolayer

per_replica_batch_size = 8

def mnist_train_dataset(global_batch_size):
    (x_train, y_train), test = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train / 255., y_train))
    dataset = dataset.shuffle(dataset.cardinality()).repeat()
    dataset = dataset.batch(global_batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return x_train.shape[0] // global_batch_size, dataset

def train_mnist(strategy, saveprefix):
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    steps_per_epoch, dataset = mnist_train_dataset(global_batch_size)
    model = twolayer.compiled_model(strategy, global_batch_size)
    model.custom_train(dataset, steps_per_epoch, 10)
    model.save_weights(f"{saveprefix}.weights.h5")

def tpu_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('local')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(topology, num_replicas=topology.num_tpus_per_task)
    strategy = tf.distribute.TPUStrategy(resolver, experimental_device_assignment=device_assignment)
    return strategy

if __name__ == "__main__":
    train_mnist(tpu_strategy(), *argv[1:])
