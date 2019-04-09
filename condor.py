"""
Train using ParameterServerStrategy on a single machine
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import numpy as np
import json
import socket
import os
import logging
import glob
import time

logging.basicConfig(level=logging.INFO)


# Import TensorFlow
import tensorflow_datasets as tfds
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

EXTENSION = 'tfcluster'


def main():
    args = _parse_args()
    task, ip_address, port = find_machine_cluster_configuration(
        args.n_workers, args.worker_idx, args.initial_port, args.final_port)
    write_machine_cluster_configuration(
        task, ip_address, port, args.worker_idx, args.comm_folder)
    tf_config = read_tf_config(args.n_workers, args.comm_folder)
    print(tf_config)
    #set_tf_config(args.n_workers, args.worker_idx)
    # train()

def _parse_args():
    parser = argparse.ArgumentParser(description='Train using ParameterServerStrategy on a single machine')
    parser.add_argument('n_workers', type=int,
                        help='Total number of workers that will be used in the training')
    parser.add_argument('worker_idx', type=int,
                        help='Index of the worker for this execution. Start from 0 wich is the parameters server')
    parser.add_argument('comm_folder', type=str,
                        help='Path to the folder were the workers will write their configurations')
    parser.add_argument('--initial_port', type=int, default=49152,
                        help='Initial port for searching a free port for communication')
    parser.add_argument('--final_port', type=int, default=65535,
                        help='Final port for searching a free port for communication')

    args = parser.parse_args()
    return args

def find_machine_cluster_configuration(n_workers, worker_idx, initial_port, final_port):
    """
    Finds the ip address of the machine and a free port for working
    """
    if worker_idx:
        task = 'worker'
    else:
        task = 'ps'
    ip_address = _get_ip_address()
    initial_port, final_port = np.linspace(initial_port, final_port, n_workers+1, dtype=np.int)[worker_idx:worker_idx+2]
    for port in range(initial_port, final_port):
        if not _is_port_in_use(port):
            return task, ip_address, port
    raise Exception('All ports are being used [%i, %i]' % (initial_port, final_port))

def _get_ip_address():
    # https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
    return [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]

def _is_port_in_use(port):
    # https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def write_machine_cluster_configuration(task, ip_address, port, worker_idx, comm_folder):
    if not os.path.exists(comm_folder):
        os.makedirs(comm_folder, exist_ok=True)
    filepath = os.path.join(comm_folder, '%s_%s_%s:%s.%s' % (task, worker_idx, ip_address, port, EXTENSION))
    with open(filepath, 'w') as f:
        json.dump(dict(task=task, worker_idx=worker_idx, ip_address=ip_address, port=port), f)

def read_tf_config(n_workers, comm_folder):
    while 1:
        conf_files = glob.glob(os.path.join(comm_folder, '*.tfcluster'))
        if len(conf_files) < n_workers:
            logging.info('Waiting for the other workers. (%i < %i)' % (len(conf_files), n_workers))
            time.sleep(1)
        else:
            break
    confs = []
    for conf_file in conf_files:
        with open(conf_file, 'r') as f:
            confs.append(json.load(f))
    tf_config = {'cluster':{}, 'task':{}}
    _add_cluster_group(tf_config, 'worker', confs)
    _add_cluster_group(tf_config, 'ps', confs)
    return tf_config

def _add_cluster_group(tf_config, group, confs):
    group_confs = [conf for conf in confs if conf['task'] == group]
    sorted_idx = np.argsort([conf['worker_idx'] for conf in group_confs])
    tf_config['cluster'][group] = []
    for i in sorted_idx:
        conf = group_confs[i]
        tf_config['cluster'][group].append('%s:%s' % (conf['ip_address'], conf['port']))


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
        dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(1000).batch(10).shuffle(1000)
        model.fit(dataset, epochs=20)
        model.evaluate(dataset)

if __name__ == '__main__':
    main()