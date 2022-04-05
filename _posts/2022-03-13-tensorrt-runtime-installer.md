---
layout: post
title: TensorRT runtime installer
author: dnchoi
date: 2022-03-13 09:45:47
categories: [tensorrt, install]
tags: [tensorrt, install]
---

# TensorRT runtime install

## 1. TensorRT runtime installation

[TensorRT Download](https://developer.nvidia.com/tensorrt)

```bash
cd ~/Downloads/

tar -xvf TensorRT-*.Ubuntu-*.x86_64-gnu.cuda-11.1.cudnn8.1

cd TensorRT-*

pip3 install python/tensorrt-7.2.3.4-cp38-none-linux_x86_64.whl
# cp38 -> your python version

pip3 install uff/uff-0.6.9-py2.py3-none-any.whl

pip3 install graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl
```

## 2. TensorRT library path
```bash
nano ~/.bashrc

'
'
'

export TRT_PATH="/home/ubuntu/Downloads/TensorRT-*.Ubuntu-*.x86_64-gnu.cuda-11.1.cudnn8.1"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/TRT_PATH
```

## 2. Checked TensorRT

> Not installed

```bash
python3

Python 3.9.5 (default, Jun  4 2021, 12:28:51) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorrt as trt
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'tensorrt'
>>> 
```

> Installed

```bash
python3

Python 3.8.12 (default, Oct 12 2021, 13:49:34) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorrt as trt
>>> 
```