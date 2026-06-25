""" custom training loop の見本 """
from os import environ
import libtpu
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ['NEXT_PLUGGABLE_DEVICE_USE_C_API'] = 'true'
environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = libtpu.get_library_path()

import tensorflow as tf

per_replica_batch_size = 8
train_epochs = 3

class CustomTrainLoopModel(tf.keras.Model):
    def __init__(self, global_batch_size, *args, **kwargs):
        self.global_batch_size = global_batch_size
        super().__init__(*args, **kwargs)

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            per_example_loss = self.compute_loss(x, y, y_pred, sample_weight)
            loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        for metric in self.metrics:
            if metric.name == 'loss': metric.update_state(per_example_loss)
            else: metric.update_state(y, y_pred, sample_weight)

    @tf.function
    def dist_train_step(self, data):
        self.distribute_strategy.run(self.train_step, (data,))

    def custom_train(self, dataset, steps_per_epoch, epochs):
        dist_dataset = self.distribute_strategy.experimental_distribute_dataset(dataset)
        for step, batch in enumerate(dist_dataset, start=1):
            self.dist_train_step(batch)
            if step % steps_per_epoch == 0:
                epoch = step // steps_per_epoch
                print(f"Epoch {epoch}:", ", ".join(f"{key} = {val:g}" for key, val in self.get_metrics_result().items()))
                if epoch == epochs: break
                self.reset_metrics()

def get_dataset(global_batch_size):
    (x_train, y_train), test = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train / 255., y_train))
    dataset = dataset.shuffle(dataset.cardinality()).repeat()
    dataset = dataset.batch(global_batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return x_train.shape[0] // global_batch_size, dataset

class TwolayerModel(CustomTrainLoopModel):
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

def create_compiled_model(strategy, global_batch_size):
    with strategy.scope():
        model = TwolayerModel(global_batch_size)
        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])
    return model

def tpu_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('local')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(topology, num_replicas=topology.num_tpus_per_task)
    return tf.distribute.TPUStrategy(resolver, experimental_device_assignment=device_assignment)

def run(train_epochs):
    strategy = tpu_strategy()
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    steps_per_epoch, dataset = get_dataset(global_batch_size)
    model = create_compiled_model(strategy, global_batch_size)
    #model.test_on_batch(*map(lambda spec: tf.zeros((1, *spec.shape.as_list()[1:]), spec.dtype), dataset.element_spec))
    model.custom_train(dataset, steps_per_epoch, train_epochs)

if __name__ == '__main__':
    run(train_epochs)
