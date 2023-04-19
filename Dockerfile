FROM nvidia/cuda:11.2.0-devel-ubuntu20.04

ENV TZ=America/Toronto
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y && \
    apt-get install -y \
        python3 \
        python3-dev \
        python3-setuptools \
        python3-pip \
        gcc \
        libtinfo-dev \
        zlib1g-dev \
        build-essential \
        cmake \
        libedit-dev \
        libxml2-dev \
        wget \
        lsb-release \
        software-properties-common
    
RUN cd /home && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 12

COPY . /home/tvm

RUN cd /home/tvm && \
    mkdir build && \
    cp cmake/config.cmake build && \
    cd build && \
    cmake .. && \
    make -j

ENV TVM_LIBRARY_PATH=/home/tvm/build
ENV TVM_HOME=/home/tvm
ENV PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

RUN pip3 install \
        tqdm \
        numpy==1.23.4 \
        decorator \
        torch==1.7.0 \
        torchvision==0.8.1 \
        scipy \
        attrs \
        matplotlib \
        psutil