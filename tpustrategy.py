from os import environ
import libtpu
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ['NEXT_PLUGGABLE_DEVICE_USE_C_API'] = 'true'
environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = libtpu.get_library_path()
import tensorflow as tf

from run_strategy import distribute_test

def tpu_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('local')
    print(f"resolver.get_tpu_system_metadata().num_hosts = {resolver.get_tpu_system_metadata().num_hosts}")
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    print(f"topology.num_tpus_per_task: {topology.num_tpus_per_task}")
    print(f"topology.missing_devices: {topology.missing_devices}")
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(topology, num_replicas=topology.num_tpus_per_task)
    strategy = tf.distribute.TPUStrategy(resolver, experimental_device_assignment=device_assignment)
    return strategy

distribute_test(tpu_strategy())
