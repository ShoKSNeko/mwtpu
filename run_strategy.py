import tensorflow as tf

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
def replica_fn(vec):
    return vec[tf.distribute.get_replica_context().replica_id_in_sync_group]

def distribute_test(strategy):
    num_replicas = strategy.num_replicas_in_sync
    vec = tf.random.uniform([num_replicas])
    print(f"distribute values: {vec.numpy()}, sum = {sum(vec)}")
    distv = strategy.run(replica_fn, (vec,))
    print(f"result:\n{distv}\ntotal = {strategy.reduce('SUM', distv, None)}")
