---
layout: post
title: ROS install in Ubuntu 
author: luke-dongnyeok
date: 2022-02-27 09:45:47
categories: [install, ROS, ubuntu]
tags: [install, ROS, ubuntu]
---

# ROS install

## 1. Setup your sources.list

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

```

## 2. Set up your keys
```bash
sudo apt install curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```

## 3. Installation
```bash
sudo apt update
```
* Desktop-Full Install: (Recommended) : ROS, rqt, rviz, robot-generic libraries, 2D/3D simulators and 2D/3D perception

```bash
sudo apt install ros-melodic-desktop-full
```

* ROS-Base: (Bare Bones) ROS package, build, and communication libraries. No GUI tools.
```bash
sudo apt install ros-melodic-ros-base
```

* Individual Package: You can also install a specific ROS package (replace underscores with dashes of the package name):
```bash
sudo apt install ros-melodic-PACKAGE
```

## 4. Environment setup
```bash
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

or 

```bash
vi ~/.bashrc

.
.
.

source /opt/ros/melodic/setup.bash


source ~/.bashrc
```

* if you use zsh
```bash
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.zshrc
```

or 

```bash
vi ~/.zshrc

.
.
.

source /opt/ros/melodic/setup.zsh


source ~/.zshrc
```

## 5. Dependencies for building packages
```bash
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential

sudo apt install python-rosdep

sudo rosdep init
rosdep update
```

## 6. CV_library install
```bash
sudo apt -y install ros-melodic-opencv*
sudo apt -y install ros-melodic-usb-cam
sudo apt -y install ros-melodic-cv-bridge
sudo apt -y install ros-melodic-cv-camera
```

## 7. ROS Installer.sh
```bash
#!/bin/bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'



sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

sudo apt update
sudo apt install ros-melodic-desktop-full -y


apt search ros-melodic

echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential -y


sudo apt install python-rosdep -y

sudo rosdep init
rosdep update

sudo apt -y install ros-melodic-opencv*
sudo apt -y install ros-melodic-usb-cam
sudo apt -y install ros-melodic-cv-bridge
sudo apt -y install ros-melodic-cv-camera
```