---
layout: post
title: Docker install in Ubuntu 
author: luke-dongnyeok
date: 2022-02-28 09:45:47
categories: [install, Docker, ubuntu]
tags: [install, Docker, ubuntu]
---

# Docker install

## 1. Install essential packages required for Docker installation

```bash
sudo apt -y install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
```

## 2. GPG Key
```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

## 3. Docker repository registered
```bash
sudo add-apt-repository \
"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) \
stable"
```

## 4. Docker install
```bash
sudo apt update && sudo apt -y install docker-ce docker-ce-cli containerd.io
```

* System boot started docker system
```bash
sudo systemctl enable docker && service docker start
```
* docker status checked
```bash
service docker status
```
```bash
● docker.service - Docker Application Container Engine
     Loaded: loaded (/lib/systemd/system/docker.service; enabled; vendor preset: enabled)
     Active: active (running) since Fri 2022-02-25 12:12:53 KST; 2 days ago
TriggeredBy: ● docker.socket
       Docs: https://docs.docker.com
   Main PID: 3472445 (dockerd)
      Tasks: 43
     Memory: 11.0G
     CGroup: /system.slice/docker.service
             ├─3472445 /usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock
             ├─3472588 /usr/bin/docker-proxy -proto tcp -host-ip 0.0.0.0 -host-port 9000 -container-ip 172.17.0.2 -container-port 9000
             └─3472597 /usr/bin/docker-proxy -proto tcp -host-ip :: -host-port 9000 -container-ip 172.17.0.2 -container-port 9000
```
---
# Portainer install
portainer is a GUI web service that helps you easily manage docker images, containers, networks, etc.

## 1. Make directory and matched volume
Host directory <-> Docker directory
```bash 
mkdir -p /data/portainer
```

## 2. Portainer install with docker
### Docker image download
```bash
docker pull portainer/portainer
```
### Docker run
```bash
docker run --name portainer -p 9000:9000 -d --restart always -v /data/portainer:/data -v /var/run/docker.sock:/var/run/docker.sock portainer/portainer
```
> --name {used docker name}<br>
> -p {port number}<br>
> -d > detached option used background<br>
> -v {volume} > connected host volume<br>
> -it > terminal option<br>
> --net {host} > host network<br>
> --privileged > host 장치 모두 접근 가능<br>
> -e {DISPLAY=unix&DISPLAY} > Docker GUI X11 window connect<br>
> docker run --option --option --option ... {docker container full name}<br><br>
> Example : Docker X11 forward & Volume connect & Network Host & Privileged mode 
```bash
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/CES2022:/workspace --net host --privileged -e DISPLAY=unix$DISPLAY --name host_ces2022 cuda-ros-ubuntu18.04-opencv440/ces2022:latest
```
![untitled.jpg](assets/img/post/docker-install/untitled.png)