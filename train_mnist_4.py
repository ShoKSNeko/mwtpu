""" マルチワーカでのトレーニング """
from sys import argv
from os import environ
import libtpu
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ['NEXT_PLUGGABLE_DEVICE_USE_C_API'] = 'true'
environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = libtpu.get_library_path()

import tensorflow as tf

from mw_util import my_worker_id, num_workers, my_hostname
import twolayer_model_2 as twolayer

per_replica_batch_size = 8

def mnist_train_dataset(per_worker_batch_size, workernum, worker_id):
    (x_train, y_train), test = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train / 255., y_train))
    dataset = dataset.repeat().shard(workernum, worker_id).shuffle(x_train.shape[0] // workernum)
    dataset = dataset.batch(per_worker_batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return x_train.shape[0] // (per_worker_batch_size * workernum), dataset

def train_mnist(strategy, grpc_address_file, saveprefix):
    hostname, workernum, worker_id = my_hostname(), num_workers(), my_worker_id()
    per_worker_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    steps_per_epoch, dataset = mnist_train_dataset(per_worker_batch_size, workernum, worker_id)
    model = twolayer.compiled_model(strategy, grpc_address_file, hostname, workernum, worker_id, per_worker_batch_size * workernum)
    model.prepare_inter_worker_comm(dataset.element_spec)
    model.custom_train(dataset, steps_per_epoch, 10)
    # セーブするのはワーカ0だけ
    if worker_id == 0: model.save_weights(f"{saveprefix}.weights.h5")
    model.synchronize_finish()

def tpu_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('local')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(topology, num_replicas=topology.num_tpus_per_task)
    strategy = tf.distribute.TPUStrategy(resolver, experimental_device_assignment=device_assignment)
    return strategy

if __name__ == "__main__":
    train_mnist(tpu_strategy(), *argv[1:])
