---
layout: post
title: ROS Launcher
author: luke-dongnyeok
date: 2022-02-27 09:45:47
categories: [ROS]
tags: [ROS]
---

# ROS Launcher

# 1. Host
* Make launch dir

```bash
cd {Workspace dir}/src/{Package name}
mkdir launch
cd launch
```

* Make package.launch file

```bash
vi {package}.launch

<?xml version="1.0" encoding="UTF-8"?>
<launch>
  <machine name="machine name" address="machine ip address" user = "machine user name" password = "paswd" env-loader="machine working dir/env.sh"/> 
  <node machine="machine name" name="package name" pkg="package name" type="package name" output="log"/>
  <node pkg="local package name" name="local package name" type="local package name" output="screen"> </node>
</launch>
```
# 2. Client

```bash
cd {Workspace dir}

vi env.sh

#!/bin/bash
source /opt/ros/melodic/setup.bash
source {Workspace absolute path}/devel/setup.bash
export ROS_HOSTNAME={Client IP}
export ROS_HOME={Workspace absolute path}
exec "$@"
```

# 3. roslaunched machine
```bash
ssh -oHostKeyAlgorithms='ssh-rsa' client@192.168.0.56
```