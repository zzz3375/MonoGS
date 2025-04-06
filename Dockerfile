FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
# Check here https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
# for for the mapping.
# E.g. AWS g5g.4xlarge uses Tesla T4G GPU, which uses sm_75 = CUDA ARCH 7.5 (pytorch has no wheels for this)
# E.g. AWS g5.4xlarge  uses      A10G GPU, which uses sm_86 = CUDA ARCH 8.6 (pytorch has wheels for this; good)
# This variable TORCH_CUDA_ARCH_LIST is needed when building the submodules
# You can find out the architecture via torch.cuda.get_device_capability(0)
ENV TORCH_CUDA_ARCH_LIST="8.6"

ENV NVIDIA_DRIVER_CAPABILITIES="all"
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y apt-utils

RUN apt-get install -y \
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
    python3-tk \
    libc6 \
    x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

HEALTHCHECK CMD nvidia-smi || exit 1

# Alias python3 -> python for convenience
RUN ln -s $(which python3) /usr/bin/python
# RUN apt-get install -y python-is-python3

RUN pip install --upgrade pip==23.3.1

# Downgrade numpy to avoid compatibility issues
RUN pip install "numpy<2"

RUN pip config set global.timeout 600
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN pip install \
    torchmetrics==1.4.1 \
    opencv-python==4.8.1.78 \
    urllib3==1.26.15 \
    chardet==4.0.0 \
    requests==2.28.2 \
    munch==4.0.0 \
    trimesh==4.4.7 \
    evo==1.31.0 \
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