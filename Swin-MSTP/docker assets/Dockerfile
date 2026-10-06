FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /model

# Ctcdecode depends on compiling C++ code. Here we install some dependencies like gcc, g++, build-essential and some other nice-to-haves that we are going to need either way. SlowFastSign depends on opencv-python for example, and opencv-python depends on ffmpeg. That's why wer'e installing it and a bunch of other libs.
RUN apt update -y
RUN apt install -y git wget ffmpeg libsm6 libxext6 gcc g++ build-essential

# Set default command for the container
CMD ["bash"]

# Download miniconda (in my experience it works better than virtualenv)
RUN mkdir -p /miniconda && wget https://repo.anaconda.com/miniconda/Miniconda3-py312_25.5.1-1-Linux-x86_64.sh -O /miniconda/miniconda.sh && bash /miniconda/miniconda.sh -b -u -p /miniconda

# Create a virtualenv and download the required software.
RUN . /miniconda/bin/activate && conda tos accept && conda tos accept --channel nvidia && conda create -n modelvenv && conda install -n modelvenv -y python=3.9 && conda install -n modelvenv -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia && conda init --all

# As soon as you enter the container, make sure to install the dependencies for ctcdecode and SlowFastSign separately. They should install flawlessly, though.
