---
title: Ubuntu 18.04 install nvidia CUDA toolkit
author: luke-dongnyeok
date: 2022-01-12 12:57:47
categories: [ubuntu, install, cuda]
tags: [install, nvidia, cuda]
---

# CUDA & cudnn driver install 

## 1. Download CUDA driver
CUDA를 설치하기 위해서는 아래의 링크에서 사용하고자 하는 CUDA version을 다운로드 한다.

[NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

|Select Target Platform||
|:---|---:|
|OS|Linux|
|Architecture|x86_64|
|Distribution|Ubuntu|
|Version|18.04|
|Installer Type|runfile [local]|

## 2. Install CUDA.runfile
```bash
cd {CUDA Download folder}
sudo chmod +x {Download CUDA file}.run
sudo ./{Download CUDA file}.run
```
다음과 같이 실행하면 아래와 같은 결과를 볼 수 있다.
```bash
┌──────────────────────────────────────────────────────────────────────────────┐
│ Existing package manager installation of the driver found. It is strongly    │
│ recommended that you remove this before continuing.                          │
│ Abort                                                                        │
│ Continue                                                                     │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│ Up/Down: Move | 'Enter': Select                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```
Continue 선택 후 Enter
```bash
┌──────────────────────────────────────────────────────────────────────────────┐
│  End User License Agreement                                                  │
│  --------------------------                                                  │
│                                                                              │
│  NVIDIA Software License Agreement and CUDA Supplement to                    │
│  Software License Agreement.                                                 │
│                                                                              │
│                                                                              │
│  Preface                                                                     │
│  -------                                                                     │
│                                                                              │
│  The Software License Agreement in Chapter 1 and the Supplement              │
│  in Chapter 2 contain license terms and conditions that govern               │
│  the use of NVIDIA software. By accepting this agreement, you                │
│  agree to comply with all the terms and conditions applicable                │
│  to the product(s) included herein.                                          │
│                                                                              │
│                                                                              │
│  NVIDIA Driver                                                               │
│                                                                              │
│                                                                              │
│──────────────────────────────────────────────────────────────────────────────│
│ Do you accept the above EULA? (accept/decline/quit):                         │
│ accept                                                                       │
└──────────────────────────────────────────────────────────────────────────────┘
```
accept 입력 후 Enter

다음 화면에서 CUDA를 설치할 항목에 대해 선택 사항들이 보인다.
```bash
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Installer                                                               │
│ - [ ] Driver                                                                 │
│      [ ] 455.23.05                                                           │
│ + [X] CUDA Toolkit 11.1                                                      │
│   [X] CUDA Samples 11.1                                                      │
│   [X] CUDA Demo Suite 11.1                                                   │
│   [X] CUDA Documentation 11.1                                                │
│   Options                                                                    │
│   Install                                                                    │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│ Up/Down: Move | Left/Right: Expand | 'Enter': Select | 'A': Advanced options │
└──────────────────────────────────────────────────────────────────────────────┘
```
만약 Nvidia Driver를 설치 했다면, Driver 부분에서 space bar를 통해 설치 항목을 제거한다.

[X] -> 선택&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;&nbsp;] -> 선택 하지 않음
 
```bash
selecte saved directory

/usr/local/cuda
```

... processing

```bash
vi ~/.bashrc
```
Edit bashrc file
```bash
.
.
.
if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}} 
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
Reboot system
```bash
sudo reboot
```
Checked installed cuda version
```bash
nvcc -V

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Tue_Sep_15_19:10:02_PDT_2020
Cuda compilation tools, release 11.1, V11.1.74
Build cuda_11.1.TC455_06.29069683_0
```

# CUDNN install

## 1. Download cudnn file
설치한 CUDA version에 맞는 cudnn lib를 아래의 링크를 통해 다운받아 준다.
cudnn file을 다운 받기 위해서는 nvidia에 sign up이 필요하다.


[NVIDIA cudnn lib Archive](https://developer.nvidia.com/cudnn).

## 2. Copyed cudnn lib file and linked
```bash
tar -xzvf cudnn-x.x-linux-x64-v8.x.x.x.tgz 
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

## 3. Checked cudnn
CUDNN version 확인을 위한 아래의 코드는 CUDA 10.0 이상일 때만 사용이 가능하다. 
```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```
