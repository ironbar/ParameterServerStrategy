# Tensorflow ParameterServerStrategy

The goal of this repository is to make a simple experiment that uses the new ParameterServerStrategy
from tensorflow.

## Useful links

https://www.tensorflow.org/alpha/guide/distribute_strategy  
https://www.tensorflow.org/alpha/tutorials/distribute/keras  
https://www.tensorflow.org/alpha/tutorials/distribute/multi_worker  

## local.py

This script launches a parameter server training in the local machine. It allows to
configure how many workers are we going to use and the index of the worker. Index 0 is for the parameter serving service.

For example if we want to use 3 workers we should launch the following commands on **3 different terminals**:

    python local.py 3 0 #this starts the parameter server
    python local.py 3 1 #this starts the first worker
    python local.py 3 2 #this starts the second worker

### Observations

* If the parameter server has not started the workers will be waiting until it starts
* If the parameter server is ready and worker 2 is not ready worker 1 will start training without waiting for worker 2
* The error at the end of the training when using 2 workers is smaller than when using 1 worker
* Currently the server does not stop when the training end. Moreover the workers show an error message at the end of the training

## custom_conf.py

This script allows to give a json file with the tensorflow configuration as input. 

It's possible to replicate the train from local.py launching the following commands on **3 different terminals**:

    python custom_conf.py tf_confs/localhost/ps.json #this starts the parameter server
    python custom_conf.py tf_confs/localhost/worker_0.json #this starts the first worker
    python custom_conf.py tf_confs/localhost/worker_1.json #this starts the second worker

### Observations

* If we define a parameter server without workers and start the workers pointing to that server it does not work
* If we define the workers with incorrect ports on the parameter server it does not work
* If each worker has only information about itself and the server only the first worker trains. However if the second worker adds information about a fake first worker then both workers train. So it seems that the index of the worker is needed.

## Condor cluster

The final step is to make this work in the condor cluster. There are two problems we have to face:

1. The ip of the machines and ports is not known in advance. It's condor who does the machine assignation.
2. The ports may be in use, so we need a method to choose the ports wisely. More than one work can land on the same machine so the method has to deal with this.

The previous experiments have shown that the server needs to know the ips of the machines and the ports before starting. The workers can have fake information but it will be better and simpler to provide them the truth. One solution for this could be to wait until all the jobs have landed on their machines and have chosen their ports. Each job can write a file with its parameters so the other jobs can create the configuration.  
The disadvantage of this method is that waiting for all the jobs could take a lot of time. In the other hand probably it's the best strategy for training.

We need to find a python library for checking if a port is available. From [stackoverflow](https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python) we get the following function that I have tested and works.

```python
import socket

def is_port_in_use(port):
    # https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
```

We also need a method for getting the IP address of the machine

```python
import socket

def get_ip_address():
    # https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
    return [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]

```