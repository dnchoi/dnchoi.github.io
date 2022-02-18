---
layout: post
title: OpenCV install in Ubuntu 
author: luke-dongnyeok
date: 2022-01-25 09:45:47
categories: [install, OpenCV, ubuntu]
tags: [install, OpenCV, ubuntu]
---

# OpenCV install

## 1. Install essential packages required for OpenCV installation
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get -y install g++
sudo apt-get -y install build-essential cmake
sudo apt-get -y install pkg-config
sudo apt-get -y install libjpeg-dev libpng-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev
sudo apt-get -y install lib41-dev v4l-utils
sudo apt-get -y install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get -y install libgtk2.0-dev
sudo apt-get -y install mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev  
sudo apt-get -y install libatlas-base-dev gfortran libeigen3-dev
sudo apt-get -y install python2.7-dev python3-dev python-numpy python3-numpy
sudo apt-get -y install libgtk-3-dev
sudo apt-get -y install libqt5-dev
```

## 2. Install gstreamer
```bash
sudo apt install -y libgstreamer1.0-0 libgstreamer1.0-dev \
gstreamer1.0-tools gstreamer1.0-doc gstreamer1.0-x gstreamer1.0-plugins-base \
gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly  \
gstreamer1.0-alsa gstreamer1.0-libav gstreamer1.0-gl gstreamer1.0-gtk3 \
gstreamer1.0-qt5 gstreamer1.0-pulseaudio libgstreamer-plugins-base1.0-dev
```

## 3. Download OpenCV SDK
```bash
mkdir -p ~/SDK_Tools/OpenCV_440
cd ~/SDK_Tools/OpenCV_440
wget -O opencv.zip https://github.com/Itseez/opencv/archive/4.4.0.zip
unzip opencv.zip
wget –O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/4.4.0.zip
mv 4.4.0.zip opencv_contrib.zip
unzip opencv_contrib.zip
cd opencv-4.4.0
mkdir build
cd build
```

## 4. CMake build
```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_C_COMPILER=/usr/bin/gcc-7 \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=ON \
-D BUILD_DOCS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PACKAGE=OFF \
-D BUILD_EXAMPLES=OFF \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.1 \
-D WITH_CUDA=ON \
-D WITH_CUBLAS=ON \
-D WITH_CUFFT=ON \
-D WITH_NVCUVID=ON \
-D WITH_IPP=OFF \
-D WITH_V4L=ON \
-D WITH_1394=OFF \
-D WITH_GTK=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_OPENCL=OFF \
-D WITH_EIGEN=ON \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D BUILD_JAVA=OFF \
-D BUILD_opencv_python3=ON \
-D BUILD_opencv_python2=OFF \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D OPENCV_SKIP_PYTHON_LOADER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.4.0/modules \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=7.5 \
-D CUDA_ARCH_PTX=7.5 \
-D CUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so.8.1.1 \
-D CUDNN_INCLUDE_DIR=/usr/local/cuda/include  ..
```

## 5. Checked CPU cores
```bash
cat /proc/cpuinfo | grep processor | wc -l
```

## 6. Make and install OpenCV SDK
```bash
sudo make -j16
sudo make install
pkg-config --modversion opencv
pkg-config -libs -cflags opencv
```