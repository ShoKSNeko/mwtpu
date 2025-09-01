import tensorflow as tf

from run_strategy import distribute_test

# CPUデバイスが4個あることにする
physical_devices = tf.config.list_physical_devices('CPU')
logical_devices = [tf.config.LogicalDeviceConfiguration()] * 4
tf.config.set_logical_device_configuration(physical_devices[0], logical_devices)

distribute_test(tf.distribute.MirroredStrategy())
