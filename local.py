"""
Train using ParameterServerStrategy on a single machine
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import numpy as np
import json

# Import TensorFlow
import tensorflow_datasets as tfds
import tensorflow as tf

tf.compat.v1.disable_eager_execution()



def main():
    args = _parse_args()
    set_tf_config(args.n_workers, args.worker_idx)
    train()


def _parse_args():
    parser = argparse.ArgumentParser(description='Train using ParameterServerStrategy on a single machine')
    parser.add_argument('n_workers', type=int,
                        help='Total number of workers that will be used in the training')
    parser.add_argument('worker_idx', type=int,
                        help='Index of the worker for this execution. Start from 0 wich is the parameters server')

    args = parser.parse_args()
    return args

def set_tf_config(n_workers, worker_idx):
    IP_ADDRS = ['localhost']*n_workers
    PORTS = np.arange(12345, 12345 + n_workers)

    if worker_idx == 0:
        task = {'type': 'ps', 'index': 0}
    else:
        task = {'type': 'worker', 'index': worker_idx-1}

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ['%s:%d' % (IP_ADDRS[w], PORTS[w]) for w in range(1, n_workers)],
            'ps': ['%s:%d' % (IP_ADDRS[w], PORTS[w]) for w in range(1)],
        },
        'task': task,
    })

def train():
    strategy = tf.distribute.experimental.ParameterServerStrategy()
    with strategy.scope():
        model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
        model.compile(loss='mse', optimizer='sgd')
        dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10).shuffle(200)
        model.fit(dataset, epochs=20)
        model.evaluate(dataset)

if __name__ == '__main__':
    main()