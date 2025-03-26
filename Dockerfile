FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV NVIDIA_DRIVER_CAPABILITIES="all"

# Add PPA for newer Mesa drivers (so we'll get OpenGL>=4.3)
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:kisak/kisak-mesa

RUN apt-get update && apt-get install -y \
    lsb-release \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    freeglut3-dev \
    mesa-utils \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxmu-dev \
    libxi-dev \
    git \
    python3 \
    python3-dev \
    python3-pip \
    libc6 \
    x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Alias python3 -> python for convenience
RUN ln -s $(which python3) /usr/bin/python
RUN pip install --upgrade pip
RUN pip config set global.timeout 600
RUN pip install torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118

RUN pip install \
    torchmetrics==1.4.1 \
    opencv-python==4.8.1.78 \
    urllib3==1.26.15 \
    chardet==4.0.0 \
    requests==2.28.2 \
    munch==4.0.0 \
    trimesh==4.4.7 \
    evo==1.11.0 \
    open3d==0.18.0 \
    imgviz==1.7.5 \
    PyOpenGL==3.1.7 \
    glfw==2.7.0 \
    PyGLM==2.7.1 \
    wandb==0.17.8 \
    lpips==0.1.4 \
    rich==13.8.0 \
    ruff==0.6.2 \
    plyfile==1.0.3

# Building the submodules requires ninja
RUN pip install ninja --upgrade

WORKDIR /MonoGS

# Alternatively, if you didn't clone the repo on the host machine, run this:
# RUN git clone https://github.com/tauzn-clock/MonoGS/ --recursive
COPY . /MonoGS/

RUN pip install submodules/diff-gaussian-rasterization
RUN pip install submodules/simple-knn
