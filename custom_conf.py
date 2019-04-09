"""
Train using ParameterServerStrategy reading the tensorflow configuration from a json file
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
    set_tf_config(args.json)
    train()

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Train using ParameterServerStrategy reading the tensorflow configuration from a json file')
    parser.add_argument('json', type=str,
                        help='Path to the json file with the configuration')
    args = parser.parse_args()
    return args

def set_tf_config(json_path):
    with open(json_path, 'r') as f:
        tf_config = json.load(f)
    os.environ['TF_CONFIG'] = json.dumps(tf_config)

def train():
    strategy = tf.distribute.experimental.ParameterServerStrategy()
    with strategy.scope():
        model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
        model.compile(loss='mse', optimizer='sgd')
        dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(1000).batch(10).shuffle(1000)
        model.fit(dataset, epochs=20)
        model.evaluate(dataset)

if __name__ == '__main__':
    main()