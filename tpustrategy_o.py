from os import environ
import libtpu
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ['NEXT_PLUGGABLE_DEVICE_USE_C_API'] = 'true'
environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = libtpu.get_library_path()

import tensorflow as tf

from run_strategy import distribute_test

def tpu_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('local')
    metadata = resolver.get_tpu_system_metadata()
    print(f"resolver.get_tpu_system_metadata().num_hosts = {metadata.num_hosts}")
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    print(f"topology.missing_devices: {topology.missing_devices}")
    print(f"topology.num_tasks: {topology.num_tasks}")
    print(f"topology.num_tpus_per_task: {topology.num_tpus_per_task}")
    strategy = tf.distribute.TPUStrategy(resolver)
    print(f"strategy.num_replicas_in_sync = {strategy.num_replicas_in_sync}")
    return strategy

print(f"tensorflow version: {tf.__version__}")
distribute_test(tpu_strategy())
