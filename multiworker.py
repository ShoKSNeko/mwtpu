""" マルチワーカ化 """

from time import sleep

from portpicker import pick_unused_port

import tensorflow as tf

class MultiworkerModel(tf.keras.Model):
    def __init__(self, grpc_address_file, hostname, workernum, worker_id, global_batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grpc_address_file = grpc_address_file
        self.hostname = hostname
        self.workernum = workernum
        self.worker_id = worker_id
        self.global_batch_size = global_batch_size
        self.cumul_loss = 0.
        self.cumul_accu = 0.

    def _get_params(self, worker_id):
        q = self.param_qs[worker_id - 1]
        @tf.function(input_signature=[])
        def f(): return q.dequeue()
        return f

    def _put_grads_metrics(self, signature):
        q = self.grads_metrics_q
        @tf.function(input_signature=signature)
        def f(*args):
            q.enqueue(args)
            return ()
        return f

    def _clnt_getp(self):
        method = getattr(self.rpcclient, f"get_params{self.worker_id}")
        def f(): return method([]).get_value()
        return f

    def _clnt_putgm(self):
        method = getattr(self.rpcclient, "put_grads_metrics")
        def f(gm): method(*gm).is_ok()
        return f

    def prepare_inter_worker_comm(self, input_spec):
        # fit() 等と同様の手順で build() する
        self.test_on_batch(*map(lambda spec: tf.zeros((1, *spec.shape.as_list()[1:]), spec.dtype), input_spec))
        if self.worker_id == 0:
            varspecs = [(v.dtype, v.shape) for v in self.trainable_variables]
            gm_specs = varspecs + [('float32', tf.TensorShape([]))] * 2
            self.param_qs = [tf.queue.FIFOQueue(1, *zip(*varspecs)) for _ in range(self.workernum - 1)]
            self.grads_metrics_q = tf.queue.FIFOQueue(self.workernum - 1, *zip(*gm_specs))
            grpc_address = f"{self.hostname}:{pick_unused_port()}"
            self.rpcserver = tf.distribute.experimental.rpc.Server.create('grpc', grpc_address)
            for w in range(1, self.workernum):
                self.rpcserver.register(f"get_params{w}", self._get_params(w))
            putgm_sig = [tf.TensorSpec(shape, dtype) for dtype, shape in gm_specs]
            self.rpcserver.register("put_grads_metrics", self._put_grads_metrics(putgm_sig))
            self.rpcserver.start()
            with tf.io.gfile.GFile(f"{self.grpc_address_file}.s", 'w') as f: f.write(grpc_address)
            tf.io.gfile.rename(f"{self.grpc_address_file}.s", self.grpc_address_file)
        else:
            self.varshapes = [v.shape for v in self.trainable_variables]
            while not tf.io.gfile.exists(self.grpc_address_file): sleep(.1)
            with tf.io.gfile.GFile(self.grpc_address_file) as f: grpc_address = f.read()
            self.rpcclient = tf.distribute.experimental.rpc.Client.create('grpc', grpc_address)
            self.get_params = self._clnt_getp()
            self.put_grads_metrics = self._clnt_putgm()

    def _synchronize_params(self):
        if self.worker_id == 0:
            for q in self.param_qs:
                q.enqueue(self.trainable_variables)
        else:
            # rpc で得られる値はトレース時にはshapeが不定になっている
            for v, val, shape in zip(self.trainable_variables, self.get_params(), self.varshapes):
                v.assign(tf.ensure_shape(val, shape))

    def _reduce_grads_metrics(self, grads_and_metrics):
        if self.worker_id != 0:
            self.put_grads_metrics(self.distribute_strategy.experimental_local_results(grads_and_metrics)[0])
        else:
            collected = self.grads_metrics_q.dequeue_many(self.workernum - 1)
            return [tf.reduce_sum(c, 0) for c in collected]

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

    def app_grads(self, gradients, collected):
        gradients = [g + c for g, c in zip(gradients, collected)]
        # apply_gradients は勾配を all_reduce するが、もともとreduceされた値なので影響はない
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    @tf.function
    def dist_train_step(self, data):
        self._synchronize_params()
        grads_and_metrics = self.distribute_strategy.run(self.get_grads_and_metrics, (data,))
        collected = self._reduce_grads_metrics(grads_and_metrics)
        if self.worker_id == 0:
            self.distribute_strategy.run(self.app_grads, (grads_and_metrics[:-2], collected[:-2]))
            loss, accu = self.distribute_strategy.experimental_local_results(grads_and_metrics[-2:])[0]
            return loss + collected[-2], accu + collected[-1]
        else: return 0., 0.

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
                if self.worker_id == 0:
                    print(f"Epoch {epoch}: loss = {avg_loss:g}, accuracy = {avg_accu:g}", flush=True)
                if epoch == epochs: break
                self.cumul_loss = 0.
                self.cumul_accu = 0.

    # 終了のタイミングを全ワーカで同期する
    def synchronize_finish(self):
        if self.worker_id == 0:
            tf.io.gfile.remove(self.grpc_address_file)
            for q in self.param_qs:
                q.enqueue(self.trainable_variables)
        else:
            self.get_params()
