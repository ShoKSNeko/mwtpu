""" keras.Model の train_step のカスタマイズのベース """

import tensorflow as tf

class CustomStepModel(tf.keras.Model):
    def __init__(self, global_batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_batch_size = global_batch_size

    @staticmethod
    def _reduce_gradients(strategy, grads):
        return [strategy.reduce(tf.distribute.ReduceOp.SUM, g, None) for g in grads]

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # loss は reduction=keras.losses.Reduction.NONE としてコンパイルする
            # 分散トレーニングでは global_batch_size が各ワーカのデータ数と異なる
            per_example_loss = self.compute_loss(x, y, y_pred, sample_weight)
            loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)
        # 分散トレーニングではワーカ0に勾配の総和を送信するのでオプティマイザに任せず自分で集計する
        # apply_gradients の中でも勾配を all_reduce しているが、既に merge_call の結果なので値は変わらない
        reduce_args = tape.gradient(loss, self.trainable_variables),
        reduced_grads = tf.distribute.get_replica_context().merge_call(self._reduce_gradients, args=reduce_args)
        self.optimizer.apply_gradients(zip(reduced_grads, self.trainable_variables))
        return self.compute_metrics(x, y, y_pred, sample_weight)
