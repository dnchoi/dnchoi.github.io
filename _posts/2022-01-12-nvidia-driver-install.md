---
title: Ubuntu 18.04 install nvidia driver 
author: dnchoi
date: 2022-01-12 09:45:47
categories: [ubuntu, install, cuda]
tags: [install, nvidia, cuda]
---

# Nvidia driver install 

## 1. Download ubuntu driver
```bash
sudo apt install -y ubuntu-drivers-common
```
## 2. Download & install nvidia graphic driver
### a. Checked recommended driver
```bash
ubuntu-drivers devices

== /sys/devices/pci0000:a0/0000:a0:00.0/0000:a1:00.0 ==
modalias : pci:v000010DEd00001E02sv000010DEsd000012A3bc03sc00i00
vendor   : NVIDIA Corporation
model    : TU102 [TITAN RTX]
driver   : nvidia-driver-470 - distro non-free recommended
driver   : nvidia-driver-460-server - distro non-free
driver   : nvidia-driver-460 - distro non-free
driver   : nvidia-driver-470-server - distro non-free
driver   : nvidia-driver-495 - distro non-free
driver   : nvidia-driver-418-server - distro non-free
driver   : nvidia-driver-450-server - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builti
```
### b. Download & install driver
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

sudo apt install -y nvidia-driver-460 
```

### c. Reboot
```bash
sudo reboot
```

### d. Checked driver
```bash
nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.106.00   Driver Version: 460.106.00   CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  TITAN RTX           Off  | 00000000:A1:00.0 Off |                  N/A |
| 41%   32C    P8     3W / 280W |      0MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  GeForce RTX 208...  Off  | 00000000:C1:00.0 Off |                  N/A |
| 35%   38C    P8    21W / 250W |      0MiB / 11018MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
