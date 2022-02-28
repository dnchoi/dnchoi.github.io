---
title: Miniconda install
author: dnchoi
date: 2022-02-04 09:45:47
categories: [python]
tags: [miniconda, install, python]
---

# Miniconda 설치

## 1. Download Miniconda3

A. Download lastest version
```bash
cd Downloads
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

## 2. Changed permision
```bash
chmod +x Miniconda3-latest-Linux-x86_64.sh
```
or
```bash
chmod 777 Miniconda3-latest-Linux-x86_64.sh
```

## 3. Run install script
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

```bash

Welcome to Miniconda3 py39_4.10.3

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>> 
===================================
End User License Agreement - Miniconda
===================================

Copyright 2015-2021, Anaconda, Inc.

All rights reserved under the 3-clause BSD License:

This End User License Agreement (the "Agreement") is a legal agreement between y
ou and Anaconda, Inc. ("Anaconda") and governs your use of Miniconda.
.
.
(중략)
.
.
You acknowledge that, as between you and Anaconda, Anaconda owns all right, titl
e, and interest, including all intellectual property rights, in and to Miniconda
 and, with respect to third-party products distributed with or through Miniconda
--More-- (Q 입력으로 Next step)

Do you accept the license terms? [yes|no]
[no] >>> yes

Miniconda3 will now be installed into this location:
/home/{Your PC name}/miniconda3

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home/{Your PC name}/miniconda3] >>>

.
.
(install.....)
.
.

  xz                 pkgs/main/linux-64::xz-5.2.5-h7b6447c_0
  yaml               pkgs/main/linux-64::yaml-0.2.5-h7b6447c_0
  zlib               pkgs/main/linux-64::zlib-1.2.11-h7b6447c_3


Preparing transaction: done
Executing transaction: done
installation finished.
WARNING:
    You currently have a PYTHONPATH environment variable set. This may cause
    unexpected behavior when running the Python interpreter in Miniconda3.
    For best results, please verify that your PYTHONPATH only points to
    directories of packages that are compatible with the Python interpreter
    in Miniconda3: /home/luke/miniconda3
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
[no] >>> yes
./Miniconda3-latest-Linux-x86_64.sh: 472: ./Miniconda3-latest-Linux-x86_64.sh: [[: not found
no change     /home/luke/miniconda3/condabin/conda
no change     /home/luke/miniconda3/bin/conda
no change     /home/luke/miniconda3/bin/conda-env
no change     /home/luke/miniconda3/bin/activate
no change     /home/luke/miniconda3/bin/deactivate
no change     /home/luke/miniconda3/etc/profile.d/conda.sh
no change     /home/luke/miniconda3/etc/fish/conf.d/conda.fish
no change     /home/luke/miniconda3/shell/condabin/Conda.psm1
no change     /home/luke/miniconda3/shell/condabin/conda-hook.ps1
no change     /home/luke/miniconda3/lib/python3.9/site-packages/xontrib/conda.xsh
no change     /home/luke/miniconda3/etc/profile.d/conda.csh
modified      /home/luke/.bashrc

==> For changes to take effect, close and re-open your current shell. <==

If you'd prefer that conda's base environment not be activated on startup, 
   set the auto_activate_base parameter to false: 

conda config --set auto_activate_base false

Thank you for installing Miniconda3!
```

## 4. Checked ~/.bashrc
```bash
vi ~/.bashrc

.
.
.
.
.
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/luke/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/$
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/luke/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/luke/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/luke/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

## 5. Conda create env
```bash
conda create -n torch python=3.8 -y
```

## 6. Conda env activate
```bash
conda activate torch

pip install numpy opencv-python matplotlib

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
[Reference-1]

## 7. Conda env freeze export
```bash
conda activate torch
pip freeze > requirements.txt
```
or
```bash
conda env export > environment.yaml
```

## 8. Conda remove env
```bash
conda env remove -n torch -y
```

## 9. Create conda env using environment.yaml
```bash
conda env create --file environment.yaml
```
이 때, 
Solving environment: failed
ResolvePackageNotFound:

에러가 뜰 수 있음.
PC1에서 설치된 라이브러리가 PC2에서 설치가 불가능한 것인데, PC의 운영체제가 다른 경우에 발생하는 것으로 보임.
해결 방법 : 수동으로 ResoevePackageNotFound 에서 출력된 리스트를 environment.yaml 파일에서 지운 후, 다시 시도.<br>
[Reference-2]

[Reference-1]: https://pytorch.org/get-started/previous-versions/ 
[Reference-2]: https://stackoverflow.com/questions/49154899/resolvepackagenotfound-create-env-using-conda-and-yml-file-on-macos