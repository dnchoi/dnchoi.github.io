---
title: Gstreamer pipeline
author: luke-dongnyeok
date: 2022-01-25 09:45:47
categories: [Gstreamer]
tags: [Pipeline, Gstreamer]
---

# Gstreamer

## 1. Checked Gstreamer devices
```bash
gst-device-monitor-1.0

---------------------------------------

Probing devices...

... (omit)

Device found:

	name  : USB 2.0 Camera: HD USB Camera
	class : Video/Source
	caps  : video/x-raw, format=(string)YUY2, width=(int)1920, height=(int)1080, pixel-aspect-ratio=(fraction)1/1, framerate=(fraction)6/1;
	        video/x-raw, format=(string)YUY2, width=(int)1280, height=(int)1024, pixel-aspect-ratio=(fraction)1/1, framerate=(fraction)6/1;
	        video/x-raw, format=(string)YUY2, width=(int)1280, height=(int)720, pixel-aspect-ratio=(fraction)1/1, framerate=(fraction)9/1;
	        video/x-raw, format=(string)YUY2, width=(int)1024, height=(int)768, pixel-aspect-ratio=(fraction)1/1, framerate=(fraction)6/1;
	        video/x-raw, format=(string)YUY2, width=(int)800, height=(int)600, pixel-aspect-ratio=(fraction)1/1, framerate=(fraction)20/1;
	        video/x-raw, format=(string)YUY2, width=(int)640, height=(int)480, pixel-aspect-ratio=(fraction)1/1, framerate=(fraction)30/1;
	        video/x-raw, format=(string)YUY2, width=(int)320, height=(int)240, pixel-aspect-ratio=(fraction)1/1, framerate=(fraction)30/1;
	        image/jpeg, width=(int)1920, height=(int)1080, pixel-aspect-ratio=(fraction)1/1, framerate=(fraction)30/1;
	        image/jpeg, width=(int)1280, height=(int)1024, pixel-aspect-ratio=(fraction)1/1, framerate=(fraction)30/1;
	        image/jpeg, width=(int)1280, height=(int)720, pixel-aspect-ratio=(fraction)1/1, framerate=(fraction)60/1;
	        image/jpeg, width=(int)1024, height=(int)768, pixel-aspect-ratio=(fraction)1/1, framerate=(fraction)30/1;
	        image/jpeg, width=(int)800, height=(int)600, pixel-aspect-ratio=(fraction)1/1, framerate=(fraction)60/1;
	        image/jpeg, width=(int)640, height=(int)480, pixel-aspect-ratio=(fraction)1/1, framerate=(fraction)61612/513;
	        image/jpeg, width=(int)320, height=(int)240, pixel-aspect-ratio=(fraction)1/1, framerate=(fraction)61612/513;
	properties:
		udev-probed = true
		device.bus_path = pci-0000:00:14.0-usb-0:11:1.0
		sysfs.path = /sys/devices/pci0000:00/0000:00:14.0/usb1/1-11/1-11:1.0/video4linux/video0
		device.bus = usb
		device.subsystem = video4linux
		device.vendor.id = 05a3
		device.vendor.name = "HD\\x20Camera\\x20Manufacturer"
		device.product.id = 9230
		device.product.name = "USB\ 2.0\ Camera:\ HD\ USB\ Camera"
		device.serial = HD_Camera_Manufacturer_USB_2.0_Camera
		device.capabilities = :capture:
		device.api = v4l2
		device.path = /dev/video0
		v4l2.device.driver = uvcvideo
		v4l2.device.card = "USB\ 2.0\ Camera:\ HD\ USB\ Camera"
		v4l2.device.bus_info = usb-0000:00:14.0-11
		v4l2.device.version = 328861 (0x0005049d)
		v4l2.device.capabilities = 2225078273 (0x84a00001)
		v4l2.device.device_caps = 69206017 (0x04200001)
	gst-launch-1.0 v4l2src ! ...

... (omit)

```

# 2. USB type camera pipeline

## Used gst-launch
```bash
gst-launch-1.0 v4l2src ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! videoconvert ! autovideosink
```
# 3. RTSP type camera pipeline

## Used gst-launch
```bash
gst-launch-1.0 rtspsrc location=uri latency=1 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! autovideosink
```

# 4. Used application (OpenCV)
## USB camera
### C++
```c++
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int ac, char** av) {
	std::string gst_cmd = "v4l2src ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! videoconvert ! appsink";
	cv::VideoCapture cap(gst_cmd);

	if (!cap.isOpened())
	{
		std::cout << "Can't open the camera" << std::endl;
		return -1;
	}

	cv::Mat img;

	while (cap.isOpened())
	{
		try
		{
			cap >> img;
			
			if (img.empty())
			{
				std::cout << "empty image" << std::endl;
				return 0;
			}
			
			cv::imshow("camera img", img);

			if (cv::waitKey(25) == 27)
				break;
		}
		catch( cv::Exception& e )
		{
			const char* err_msg = e.what();
			std::cout << "exception caught: " << err_msg << std::endl;
		}
	}

	return 0;
}
```
### Python
```python
import cv2
import sys

gst_cmd = "v4l2src ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! videoconvert ! appsink"

cap = cv2.VideoCapture(gst_cmd)

if !cap.isOpened():
	print("Can't open the camera")
	sys.exit(0)

while cap.isOpened():
	try:
		ret, frame = cap.read()
		if ret:
			cv2.imshow('video', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break

	except Exception as ex:
		print('OpenCV except Error : ', ex)

cap.release()
cv2.destroyAllWindows()
```

## RTSP camera
### C++
```c++
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int ac, char** av) {
	std::string gst_cmd = "rtspsrc location=rtsp://192.168.0.11:5545/live1 latency=1 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink";

	cv::VideoCapture cap(gst_cmd);

	if (!cap.isOpened())
	{
		std::cout << "Can't open the camera" << std::endl;
		return -1;
	}

	cv::Mat img;

	while (cap.isOpened())
	{
		try
		{
			cap >> img;
			
			if (img.empty())
			{
				std::cout << "empty image" << std::endl;
				return 0;
			}
			
			cv::imshow("camera img", img);

			if (cv::waitKey(25) == 27)
				break;
		}
		catch( cv::Exception& e )
		{
			const char* err_msg = e.what();
			std::cout << "exception caught: " << err_msg << std::endl;
		}
	}

	return 0;
}
```
### Python
```python
import cv2
import sys

gst_cmd = "rtspsrc location=rtsp://192.168.0.11:5545/live1 latency=1 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink"

cap = cv2.VideoCapture(gst_cmd)

if !cap.isOpened():
	print("Can't open the camera")
	sys.exit(0)

while cap.isOpened():
	try:
		ret, frame = cap.read()
		if ret:
			cv2.imshow('video', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break

	except Exception as ex:
		print('OpenCV except Error : ', ex)

cap.release()
cv2.destroyAllWindows()
```