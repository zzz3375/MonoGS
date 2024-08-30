FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY :0
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
ENV NVIDIA_DRIVER_CAPABILITIES="all"

RUN apt-get update && apt-get install -y \
    lsb-release \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    freeglut3-dev \
    mesa-utils \
    libxmu-dev \
    libxi-dev \
    git \
    python3 \
    python3-dev \
    python3-pip \
    libc6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/tauzn-clock/MonoGS/ --recursive
WORKDIR /MonoGS
RUN pip install --upgrade pip
RUN pip install -r requirement.txt

RUN pip install submodules/diff-gaussian-rasterization
RUN pip install submodules/simple-knn
