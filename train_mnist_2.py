""" 単一ワーカでのトレーニング """
""" カスタマイズした train_step """
from sys import argv
from os import environ
import libtpu
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ['NEXT_PLUGGABLE_DEVICE_USE_C_API'] = 'true'
environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = libtpu.get_library_path()

import tensorflow as tf

import twolayer_model_1 as twolayer

batch_size = 32

def mnist_train_dataset():
    (x_train, y_train), test = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train / 255., y_train))
    dataset = dataset.shuffle(dataset.cardinality())
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train_mnist(strategy, global_batch_size, saveprefix):
    dataset = mnist_train_dataset()
    model = twolayer.compiled_model(strategy, global_batch_size)
    model.fit(dataset, epochs=10)
    model.save_weights(f"{saveprefix}.weights.h5")

def tpu_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('local')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(topology, num_replicas=topology.num_tpus_per_task)
    strategy = tf.distribute.TPUStrategy(resolver, experimental_device_assignment=device_assignment)
    return strategy

if __name__ == "__main__":
    train_mnist(tpu_strategy(), batch_size, *argv[1:])
