FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY :0

RUN apt-get update && apt-get install -y \
    lsb-release \
    git \
    python3 \
    python3-dev \
    python3-pip \
    libc6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/tauzn-clock/MonoGS/ --recursive
WORKDIR /MonoGS
RUN pip3 install -r requirement.txt
