---
layout: post
title: ONNX runtime example
author: dnchoi
date: 2022-03-01 09:45:47
categories: [onnx, python, c++]
tags: [onnx, python, c++]
---

# ONNX Runtime Example

## 1. Export ONNX
### pytorch
* static batch size

고정된 batch size의 onnx모델로 변환하는 방법은 input tensor의 shape을 넣어줄 때 원하는 size의 batch를 설정해서 export해주면 된다.

모델은 기본적으로 pytorch에서 제공해 주는 resnet18을 load했고, 생성될 onnx 모델의 이름, input과 output 이름, 그리고 input tensor를 원하는 shape로 생성하여 설정했다.

```python
import os
import torch
import torchvision.models as models
from loguru import logger

net = models.resnet18(pretrained=True)
net.eval()
logger.info('Finished loading model!')
device = torch.device("cuda:0")
net = net.to(device)

output_onnx = 'static.onnx'
input_names = ["input_0"]
output_names = ["output_0"]

inputs = torch.randn(2, 3, 256, 256).to(device)

torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False, 
                               input_names=input_names, output_names=output_names, 
                               opset_version=11)
```
![static](/assets/img/post/onnx-runtime/static.png)

* dynamic batch size

dynamic batch로 설정하고 싶으면 export모델을 사용할 때 option에 dynamic_axes를 설정해 주면 된다.

 

dynamic_axes를 설정해 주는 것 말고 위와 같기 때문에 모델과 output 이름 등등은 같게 설정하였다.

```python
import os
import argparse
import torch
import torchvision.models as models
from loguru import logger

net = models.resnet18(pretrained=True)
net.eval()
logger.info('Finished loading model!')
device = torch.device("cuda:0")
net = net.to(device)

output_onnx = 'dynamic.onnx'
input_names = ["input_0"]
output_names = ["output_0"]
inputs = torch.randn(1, 3, 256, 256).to(device)
logger.info(net(inputs))

dynamic_axes = {'input_0' : {0 : 'batch_size'},
                    'output_0' : {0 : 'batch_size'}}

torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                               input_names=input_names, output_names=output_names, 
                               opset_version=11, dynamic_axes = dynamic_axes)

logger.info(torch_out)
```
![dynamic](/assets/img/post/onnx-runtime/dynamic.png)


### tensorflow - todo

### keras - todo

## 2. Inference ONNX-Runtime engine

### Inference

* PYTHON

```python
import torch
import torchvision.transforms as transforms

import onnx
import onnxruntime
import os
from loguru import logger
import cv2

from glob import glob
from tqdm import tqdm 

logger.level("DEBUG")

@logger.catch()
class verify_face():
    def __init__(self, model):
        logger.trace("pytorch: {}".format(torch.__version__))
        logger.trace("onnxruntime: {}".format(onnxruntime.__version__))
        logger.trace("onnx: {}".format(onnx.__version__))
        model_path = model
        
        logger.trace("model file is exits : {}".format(os.path.isfile(model_path)))
        onnx_model = onnx.load(model_path)
        logger.trace("onnx model check : {}".format(onnx.checker.check_model(onnx_model)))

        self.ort_session = onnxruntime.InferenceSession(model_path)
        
        self.to_tensor = transforms.ToTensor()
        
    def _resize(self, img, model_size):
        _shape = img.shape
        re = None
        if model_size != _shape:
            re = cv2.resize(img, dsize=model_size[0:2])
        else:
            re = img
        return re
    
    def _tensor(self, img):
        img = self._resize(img, (112, 112, 3))
        output = self.to_tensor(img)
        output.unsqueeze_(0)
        logger.trace(output.shape)
        output = self.to_numpy(output)
        ort_inputs = {self.ort_session.get_inputs()[0].name: output}
        return ort_inputs
    
    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    def do_inference(self, img):
        ort_outs = self.ort_session.run(None, self._tensor(img))    
        
        return ort_outs[0]



def foo(verify, img):
    img_out = verify.do_inference(img)
    logger.info(img_out.shape)


if __name__ == "__main__":
    model_path = "model.onnx"
    _verify = verify_face(model_path)
    file_path = "./DATASET/LFW/lfw-py/lfw_funneled"
    
    files = glob(os.path.join(file_path, "*", "*"))
    for i in tqdm(files):    
        img = cv2.imread(i)
        foo(_verify, img)
```

* C++

## Directory Hierarchy
```bash
|—— 12.png
|—— CMakeLists.txt
|—— build_run.sh
|—— example.cpp
|—— libs
|    |—— configparser
|        |—— CMakeLists.txt
|        |—— configparser.cpp
|        |—— include
|            |—— configparser.h
|    |—— onnxruntime
|        |—— CMakeLists.txt
|        |—— frvf_onnx.cpp
|        |—— include
|            |—— frvf_onnx.h
|    |—— spdlog
|        |—— include
|            |—— spdlog
|            |—— .
|            |—— .
|            |—— .
|            |—— .
|—— main.cpp
|—— model.onnx
|—— test.ini
```

> Build
>> build_run.sh
```bash
#!/bin/sh

# exit on first error
set -e
rm -rf build
mkdir -p build
cd build

# Generate a Makefile for GCC (or Clang, depanding on CC/CXX envvar)
cmake ..

# Build (ie 'make')
# cmake --build .
make all
cd ..

./build/main

```

> CMakeFiles

>> CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.13)

# 프로젝트 정보
project(
  main
  VERSION 0.1
  DESCRIPTION "Face recognition verify test"
  LANGUAGES CXX
)
add_compile_definitions(_DEBUG_)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_FLAGS -pthread)
set(CMAKE_CXX_STANDARD 14)

message(STATUS CMAKE_BUILD_TYPE)                     # -- CMAKE_BUILD_TYPE
message(STATUS ${CMAKE_BUILD_TYPE})                  # -- Debug
message(STATUS "Configuration: ${CMAKE_BUILD_TYPE}") # -- Configuration: Debug
message(STATUS "Compiler")
message(STATUS " - ID       \t: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS " - Version  \t: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS " - Path     \t: ${CMAKE_CXX_COMPILER}\n\n")


set (onnx_Header "${CMAKE_CURRENT_SOURCE_DIR}/libs/onnxruntime/include")
set (logger_Header "${CMAKE_SOURCE_DIR}/libs/spdlog/include")
set (config_Header "${CMAKE_SOURCE_DIR}/libs/configparser/include")

message("${onnx_Header}")
message("${logger_Header}")
message("${config_Header}")

include_directories(
    ${onnx_Header}
    ${logger_Header}
    ${config_Header}
)

find_package(OpenCV REQUIRED)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

#INIT
# Lib 등록을 위한 작업 -> 해당 Lib Cmake 동작 후 build
add_subdirectory(
    libs/onnxruntime
)
message("ONNX runtime lib")
add_subdirectory(
    libs/configparser
)
message("configparser lib")
# add_subdirectory(
#     lib/ThreadPool
# )
# message("Thread pool shared lib build done")
# #INIT

add_executable (${PROJECT_NAME} main.cpp)

# include lib file in program
target_link_libraries(
    ${PROJECT_NAME}
    frvf_onnx
    configparser
    pthread
)
target_include_directories(
    ${PROJECT_NAME} 
    PUBLIC 
        ${onnx_Header} 
        ${logger_Header}
        ${config_Header}
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

```

>> libs/onnxruntime/CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.13)
project(
  frvf_onnx
  VERSION 0.1
  LANGUAGES CXX
)
message("@@ frvf_onnx CMake Start @@")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 14)
message("This lib name : ${PROJECT_NAME}")

include_directories(include)

add_library(${PROJECT_NAME} SHARED frvf_onnx.cpp)
message("ONNX RUNTIME INCLUDE DIRS : ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}")
message("OPENCV INCLUDE DIRS : ${OpenCV_INCLUDE_DIRS}")
message("ONNX RUNTIME LIB : ${ONNX_RUNTIME_LIB}")
message("OpenCV LIBRARIES : ${OpenCV_LIBRARIES}")
target_include_directories(${PROJECT_NAME} PUBLIC ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

message("@@ frvf_onnx CMake Finish @@\n\n")

```

> Code

>> src/main.cpp

```c++
#include "iostream"
#include "frvf_onnx.h"
#include <spdlog/spdlog.h>
#include "configparser.h"

// #include "spdlog/sinks/basic_file_sink.h"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG


int main(int argc, char* argv[]){
    /**
     * @brief Construct a new frvf onnx::frvf onnx object and get Instence
     * @arg model_path, useCUDA, optimizer
     * @param model_path type string 
     * @param useCUDA type bool 
     * @param optimizer type int 
     * 0 = ORT_DISABLE_ALL : To disable all optimizations
     * 1 = ORT_ENABLE_BASIC : To enable basic optimizations (Such as redundant node removals) 
     * 2 = ORT_ENABLE_EXTENDED : To enable extended optimizations(Includes level 1 + more complex optimizations like node fusions)
     * 3 = ORT_ENABLE_ALL : To Enable All possible optimizations
     */
    
    // auto file_logger = spdlog::basic_logger_mt("basic_logger", "logs/basic.txt");
    // spdlog::set_default_logger(file_logger);
    struct Arg
    {
        int _B;
        int _W;
        int _H;
        int _C;
        int _iter;
        int _acc;
        int _opti;
        std::string _model;
        std::string _engine;
    };
    
    Arg args;
    CConfigParser config("test.ini");
	if (config.IsSuccess()) {
        args._B = config.GetInt("B");
        args._W = config.GetInt("W");
        args._H = config.GetInt("H");
        args._C = config.GetInt("C");
        args._iter = config.GetInt("ITERATION");
        args._acc = config.GetInt("ACCELERATOR");

        args._model = config.GetString("MODEL");
        args._engine = config.GetString("ENGINE");
	}
    
    SPDLOG_INFO("batch size : {}", args._B);
    SPDLOG_INFO("input width : {}", args._W);
    SPDLOG_INFO("input height : {}", args._H);
    SPDLOG_INFO("input channel : {}", args._C);
    SPDLOG_INFO("iteration number : {}", args._iter);
    SPDLOG_INFO("accelerator : {}", args._acc);
    SPDLOG_INFO("model path : {}", args._model);
    SPDLOG_INFO("optimizer : {}", args._engine);
    if(args._engine == "onnx"){
        onnx_frvf::frvf_onnx *onnx;
        onnx = new onnx_frvf::frvf_onnx("model.onnx", true, 0);
        std::vector<float> result;
        float avg_ms = 0.0;
        for(int i = 0; i < 1000; i++){
            result.push_back(onnx->do_inference("12.png"));
        }
        for(int q = 0; q < result.size(); q++)
        {
            avg_ms += result[q];
        }
        avg_ms = avg_ms / 1000.0;
        SPDLOG_CRITICAL("{:03.8f}", avg_ms);
    }
    else{
        return 0;
    }
	return 0;
}
```
>> libs/onnxruntime/include/frvf_onnx.h
```h
#ifndef __FRVF_ONNX_H__
#define __FRVF_ONNX_H__

#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include "spdlog/spdlog.h"

namespace onnx_frvf{
    
    class frvf_onnx
    {
    private:
        Ort::Env *env;
        Ort::Session *sess;
        Ort::SessionOptions *sessionOptions;
        Ort::AllocatorWithDefaultOptions *allocator;
        GraphOptimizationLevel optimizer_selector(int expression);
        size_t numInputNodes;
        const char* inputName;
        std::vector<int64_t> inputDims;
        size_t numOutputNodes;
        const char* outputName;
        std::vector<int64_t> outputDims;
        
        std::vector<const char*> inputNames;
        std::vector<const char*> outputNames;
        // std::vector<Ort::Value> inputTensors;
        // std::vector<Ort::Value> outputTensors;

    public:
        frvf_onnx(std::string file_path, bool useCUDA, int OPT_OPTION);
        ~frvf_onnx();

        void _Instance(std::string file_path, bool useCUDA, int OPT_OPTION);
        float do_inference(std::string imageFilepath);
    };

}

#endif // __FRVF_ONNX_H__
```
>> libs/onnxruntime/frvf_onnx.cpp
```cpp
#include <frvf_onnx.h>

using namespace onnx_frvf;

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}


std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

/**
 * @brief Construct a new frvf onnx::frvf onnx object
 * 
 * @param file_path 
 * @param useCUDA 
 * @param OPT_OPTION 
 */
frvf_onnx::frvf_onnx(std::string file_path, bool useCUDA, int OPT_OPTION){
    
    this->_Instance(file_path, useCUDA, OPT_OPTION);
}

GraphOptimizationLevel frvf_onnx::optimizer_selector(int expression){
    GraphOptimizationLevel a;
    switch (expression)
    {
    case 0:
        a = GraphOptimizationLevel::ORT_DISABLE_ALL;
        break;
    case 1:
        a = GraphOptimizationLevel::ORT_ENABLE_BASIC;
        break;
    case 2:
        a = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
        break;
    case 3:
        a = GraphOptimizationLevel::ORT_ENABLE_ALL;
        break;
    default:
        a = GraphOptimizationLevel::ORT_DISABLE_ALL;
        break;
    }

    return a;
}

void frvf_onnx::_Instance(std::string file_path, bool useCUDA, int OPT_OPTION)
{
#ifdef _DEBUG_
    std::string modelFilepath = file_path;
    std::string instanceName{"ONNX-face-recognition"};
    sessionOptions = new Ort::SessionOptions;
    sessionOptions->SetIntraOpNumThreads(1);
    if (useCUDA)
    {
        OrtCUDAProviderOptions cuda_options{0};
        sessionOptions->AppendExecutionProvider_CUDA(cuda_options);
    }
    env = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());

    sessionOptions->SetGraphOptimizationLevel(optimizer_selector(OPT_OPTION));
    sess = new Ort::Session(*env, modelFilepath.c_str(), *sessionOptions);
    allocator = new Ort::AllocatorWithDefaultOptions;
    numInputNodes = sess->GetInputCount();
    numOutputNodes = sess->GetOutputCount();
    inputName = sess->GetInputName(0, *allocator);
    Ort::TypeInfo inputTypeInfo = sess->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    inputDims = inputTensorInfo.GetShape();
    outputName = sess->GetOutputName(0, *allocator);
    Ort::TypeInfo outputTypeInfo = sess->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    outputDims = outputTensorInfo.GetShape();

    SPDLOG_INFO("Number of Input Nodes : {}",numInputNodes);
    SPDLOG_INFO("Number of Output Nodes : {}",numOutputNodes);
    SPDLOG_INFO("Input Name : {}",inputName);
    SPDLOG_INFO("Input Type : {}",inputType);
    SPDLOG_INFO("Input Dimensions : {} {} {} {}",inputDims[0],inputDims[1],inputDims[2],inputDims[3]);
    SPDLOG_INFO("Output Name : {}",outputName);
    SPDLOG_INFO("Output Type : {}",outputType);
    SPDLOG_INFO("Output Dimensions : {} {} {} {}",outputDims[0],outputDims[1],outputDims[2],outputDims[3]);

    inputNames.push_back(inputName);
    outputNames.push_back(outputName);

#else
#endif
}

float frvf_onnx::do_inference(std::string imageFilepath){
    cv::Mat imageBGR2= cv::Mat::zeros(1, 1, CV_64F);
    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(imageBGR, resizedImageBGR,
               cv::Size(inputDims.at(2), inputDims.at(3)),
               cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(preprocessedImage.begin<float>(),
                             preprocessedImage.end<float>());
                             
    size_t outputTensorSize = vectorProduct(outputDims);
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;
    
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
        inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputDims.data(), outputDims.size()));
   
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    sess->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);

    for (int _i=0; _i < outputDims[1]; _i ++)
    {
        SPDLOG_TRACE("{}",outputTensorValues.at(_i));
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    float processtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    return processtime;
}

frvf_onnx::~frvf_onnx(){
    
}
```
