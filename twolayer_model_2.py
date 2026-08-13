""" 2層パーセプトロン """
""" マルチワーカ対応 """

import tensorflow as tf

from multiworker import MultiworkerModel

class TwolayerModel(MultiworkerModel):
    def __init__(self, grpc_address_file, hostname, workernum, worker_id, global_batch_size):
        super().__init__(grpc_address_file, hostname, workernum, worker_id, global_batch_size)
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.logits = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.hidden(x)
        x = self.dropout(x, training=training)
        return self.logits(x)

def compiled_model(strategy, grpc_address_file, hostname, workernum, worker_id, global_batch_size):
    with strategy.scope():
        model = TwolayerModel(grpc_address_file, hostname, workernum, worker_id, global_batch_size)
        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        model.compile(optimizer=optimizer, loss=loss)
    return model
