---
layout: post
title: Symbolic Link in ubuntu
author: dnchoi
date: 2022-06-14 09:45:47
categories: [etc]
tags: [etc]
---
# Symbolic link in Ubuntu

## Create symbolic link
```bash
ln -s <TARGET> <LINK_NAME>
```

* Ex)
    * libexample.so.3.1.0 파일에 대한 심볼릭 링크로 libexample.so 를 생성하고함
    ```bash
    ln -s libexample.so.3.1.0 libexample.so
    ```
    ```bash
    ln --symbolic libexample.so.3.1.0 libexample.so
    ```


## Remove symbolic link
```bash
rm <LINK_NAME>
```
* Ex)
    ```bash
    rm libexample.so
    ```

## Checked symbolic link

### ls commend
```bash
ls -l
```


### stat commend
```bash
stat <FILE>
```