---
layout: post
title: ONNX runtime install
author: dnchoi
date: 2022-03-01 09:45:47
categories: [onnx, install]
tags: [onnx, install]
---

# ONNX runtime install

## 1. Install essential packages required for ONNX runtime installation

```bash
sudo apt update && sudo apt upgrade -y && sudo apt install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    autoconf \
    automake \
    libtool \
    pkg-config \
    ca-certificates \
    wget \
    git \
    curl \
    libjpeg-dev \
    libpng-dev \
    language-pack-en \
    locales \
    locales-all \
    python3 \
    python3-py \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-pytest \
    python3-setuptools \
    libprotobuf-dev \
    protobuf-compiler \
    zlib1g-dev \
    swig \
    vim \
    gdb \
    valgrind \
    libsm6 \
    libxext6 \
    libxrender-dev \
    cmake \
    unzip
```

## 2. Install and Build ONNX runtime

* Install python lib dependency
``` bash
pip install pytest==6.2.1
pip install onnx==1.10.1
```

* Build ONNX runtime

> onnxruntime_build.sh

``` bash
#!/bin/bash
ONNXRUNTIME_VERSION=1.8.2
NUM_JOBS=16

git clone --recursive --branch v${ONNXRUNTIME_VERSION} https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
./build.sh \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr/lib/x86_64-linux-gnu/ \
    --use_cuda \
    # --use_tensorrt \
    # --tensorrt_home /usr/lib/x86_64-linux-gnu/ \
    --config RelWithDebInfo \
    --build_shared_lib \
    --build_wheel \
    --skip_tests \
    --parallel ${NUM_JOBS}

cd build/Linux/RelWithDebInfo
make install
pip install dist/*
```

