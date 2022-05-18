---
layout: post
title: Python make tar file
author: dnchoi
date: 2022-05-18 09:45:47
categories: [etc]
tags: [etc]
---
# Make tar file module with Python

## 작성 개요
작업 공간 내에 여러 파일을 압출 할 경우 Python glob 모듈을 사용하여 여러 폴더를 각자 폴더명에 맞게 tar 압축 진행

## Script

### filename.tar
```python
import tarfile
import os
from glob import glob
from tqdm import tqdm


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def main():
    path = "/opt/files_directory"
    dirss = glob(os.path.join(path, "*"))
    for i in tqdm(dirss):
        make_tarfile(i + ".tar", i)


if __name__ == "__main__":
    main()
```
### filename.tar.gz
```python
import tarfile
import os
from glob import glob
from tqdm import tqdm


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def main():
    path = "/opt/files_directory"
    dirss = glob(os.path.join(path, "*"))
    for i in tqdm(dirss):
        make_tarfile(i + ".tar", i)


if __name__ == "__main__":
    main()
```