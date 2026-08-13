""" custom training loop の見本 """

import tensorflow as tf

class CustomTrainLoopModel(tf.keras.Model):
    def __init__(self, global_batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_batch_size = global_batch_size
        self.cumul_loss = 0.
        self.cumul_accu = 0.

    def get_grads_and_metrics(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # loss は reduction=keras.losses.Reduction.NONE を指定
            # global_batch_size は分散トレーニングの場合全ワーカを合計した値とする
            per_example_loss = self.compute_loss(x, y, y_pred)
            loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)
        gradients = tape.gradient(loss, self.trainable_variables)
        accuracy = tf.reduce_sum(tf.keras.metrics.sparse_categorical_accuracy(y, y_pred))
        # 分散トレーニングではワーカ0に勾配の総和を送信するのでここでレプリカ毎の値を集計する
        return tf.distribute.get_replica_context().all_reduce(tf.distribute.ReduceOp.SUM, (*gradients, loss, accuracy))

    def app_grads(self, gradients):
        # apply_gradients は勾配を all_reduce するが、もともとreduceされた値なので影響はない
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    @tf.function
    def dist_train_step(self, data):
        grads_and_metrics = self.distribute_strategy.run(self.get_grads_and_metrics, (data,))
        self.distribute_strategy.run(self.app_grads, (grads_and_metrics[:-2],))
        return self.distribute_strategy.experimental_local_results(grads_and_metrics[-2:])[0]

    def custom_train(self, dataset, steps_per_epoch, epochs):
        dist_dataset = self.distribute_strategy.experimental_distribute_dataset(dataset)
        for step, batch in enumerate(dist_dataset, start=1):
            loss, accuracy = self.dist_train_step(batch)
            self.cumul_loss += loss
            self.cumul_accu += accuracy
            if step % steps_per_epoch == 0:
                epoch = step // steps_per_epoch
                avg_loss = self.cumul_loss / steps_per_epoch
                avg_accu = self.cumul_accu / (steps_per_epoch * self.global_batch_size)
                print(f"Epoch {epoch}: loss = {avg_loss:g}, accuracy = {avg_accu:g}", flush=True)
                if epoch == epochs: break
                self.cumul_loss = 0.
                self.cumul_accu = 0.
