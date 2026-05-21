""" 2層パーセプトロン """
""" train_step をカスタム """

import tensorflow as tf

from custom_step import CustomStepModel

class TwolayerModel(CustomStepModel):
    def __init__(self, global_batch_size):
        super().__init__(global_batch_size)
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.logits = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.hidden(x)
        x = self.dropout(x, training=training)
        return self.logits(x)

def compiled_model(strategy, global_batch_size):
    with strategy.scope():
        model = TwolayerModel(global_batch_size)
        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model
