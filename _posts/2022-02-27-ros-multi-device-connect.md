---
layout: post
title: ROS multi device connect
author: dnchoi
date: 2022-02-27 09:45:47
categories: [ROS]
tags: [ROS]
---

# ROS Multi-device connected setting

## 1. Host IP and Client IP check
* Host

```bash
ifconfig

enp7s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 192.168.0.55  netmask 255.255.255.0  broadcast 192.168.0.55
        inet6 fc00::642:1aff:fe9a:4faf  prefixlen 64  scopeid 0x0<global>
        inet6 fe80::642:1aff:fe9a:4faf  prefixlen 64  scopeid 0x20<link>
        ether 04:42:1a:9a:4f:af  txqueuelen 1000  (Ethernet)
        RX packets 936613605  bytes 318059576117 (318.0 GB)
        RX errors 0  dropped 193585  overruns 0  frame 0
        TX packets 1024147669  bytes 325361687105 (325.3 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```

* Client

```bash
ifconfig

eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 192.168.1.109  netmask 255.255.255.0  broadcast 192.168.1.255
        inet6 fc00::e6b5:f1d1:5649:da40  prefixlen 64  scopeid 0x0<global>
        inet6 fe80::2225:b6aa:8dbd:b3fb  prefixlen 64  scopeid 0x20<link>
        ether 48:b0:2d:2f:8a:ae  txqueuelen 1000  (Ethernet)
        RX packets 24629  bytes 3455618 (3.4 MB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 563  bytes 58590 (58.5 KB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
        device interrupt 37
```

## 2. Setting .bashrc
* Host

```bash
.
.
.

export ROS_HOME=$PWD
export ROS_IP=192.168.0.55
export ROS_MASTER_URI=http://192.168.0.55:11311
export ROS_HOSTNAME=$ROS_IP

source ~/.bashrc
```
* Client

```bash
.
.
.

export ROS_IP=192.168.0.56
export ROS_MASTER_URI=http://192.168.0.55:11311

source ~/.bashrc
```
