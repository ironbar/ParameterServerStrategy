FROM python:latest
RUN pip install tensorflow==2.0.0-alpha0 tensorflow_datasets
CMD ["/bin/bash"]