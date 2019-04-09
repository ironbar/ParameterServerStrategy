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