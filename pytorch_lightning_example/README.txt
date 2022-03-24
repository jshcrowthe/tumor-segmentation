This is a comparative example to show how pytorch lightning and torchio can be used for medical segmentation tasks.

basic_example contains a notebook for running the training script the way it was written by torchio.

basic_pytorch_lightning contains a notbook, model, and dataloader for running the same training in pytorch lighting.

docker has the docker container and conda yml for running both notebooks in an nvidia cuda docker container for reproducability. 

build the docker container , and modify the following command to run the container on your system:

docker run --gpus all -i -t -d --shm-size=256m -v/home/ludauter/Documents:/home/files  -p 8855:8855/tcp  project:latest /bin/bash

I use visual studio code for interfacing with the container.

Ping me if you have questions.