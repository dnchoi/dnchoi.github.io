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

>Tree

```bash
.
├── CMakeLists.txt
├── data
│   ├── images
│   │   └── european-bee-eater-2115564_1920.jpg
│   ├── labels
│   │   └── synset.txt
│   └── models
│       ├── README.md
│       └── squeezenet1.1-7.onnx
└── src
    ├── CMakeLists.txt
    └── inference.cpp
```

> Build

```bash
mkdir build; cd build

cmake ..; make all

cd src

./inference --use_cpu
./inference --use_cuda
```

> CMakeFiles

>> CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Inference VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_executable(inference inference.cpp)
target_include_directories(inference PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})

```

>> src/CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Examples VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OpenCV REQUIRED)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

add_subdirectory(src)

```

> Code

>> src/inference.cpp

```c++

// https://github.com/microsoft/onnxruntime/blob/v1.8.2/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h
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

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
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

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
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
 * @brief 
 * 
 * @param labelFilepath 
 * @return std::vector<std::string> 
 */
std::vector<std::string> readLabels(std::string& labelFilepath)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelFilepath);
    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
    return labels;
}

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char* argv[])
{
    bool useCUDA{true};
    const char* useCUDAFlag = "--use_cuda";
    const char* useCPUFlag = "--use_cpu";
    if (argc == 1)
    {
        useCUDA = false;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) == 0))
    {
        useCUDA = true;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCPUFlag) == 0))
    {
        useCUDA = false;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) != 0))
    {
        useCUDA = false;
    }
    else
    {
        throw std::runtime_error{"Too many arguments."};
    }

    if (useCUDA)
    {
        std::cout << "Inference Execution Provider: CUDA" << std::endl;
    }
    else
    {
        std::cout << "Inference Execution Provider: CPU" << std::endl;
    }

    std::string instanceName{"funzin-face-recognition"};
    std::string modelFilepath{"../data/models/squeezenet1.1-7.onnx"};
    std::string imageFilepath{
        "../data/images/european-bee-eater-2115564_1920.jpg"};
    std::string labelFilepath{"../data/labels/synset.txt"};

    std::vector<std::string> labels{readLabels(labelFilepath)};

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    if (useCUDA)
    {
        // Using CUDA backend
        // https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L329
        OrtCUDAProviderOptions cuda_options{0};
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals) 
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    const char* inputName = session.GetInputName(0, allocator);
    std::cout << "Input Name: " << inputName << std::endl;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::cout << "Input Type: " << inputType << std::endl;

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    std::cout << "Input Dimensions: " << inputDims << std::endl;

    const char* outputName = session.GetOutputName(0, allocator);
    std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::cout << "Output Type: " << outputType << std::endl;

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    std::cout << "Output Dimensions: " << outputDims << std::endl;

    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(imageBGR, resizedImageBGR,
               cv::Size(inputDims.at(2), inputDims.at(3)),
               cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    // Normalization per channel
    // Normalization parameters obtained from
    // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, resizedImage);
    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(preprocessedImage.begin<float>(),
                             preprocessedImage.end<float>());

    size_t outputTensorSize = vectorProduct(outputDims);
    assert(("Output tensor size should equal to the label set size.",
            labels.size() == outputTensorSize));
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
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

    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);

    int predId = 0;
    float activation = 0;
    float maxActivation = std::numeric_limits<float>::lowest();
    float expSum = 0;
    for (int i = 0; i < labels.size(); i++)
    {
        activation = outputTensorValues.at(i);
        expSum += std::exp(activation);
        if (activation > maxActivation)
        {
            predId = i;
            maxActivation = activation;
        }
    }
    std::cout << "Predicted Label ID: " << predId << std::endl;
    std::cout << "Predicted Label: " << labels.at(predId) << std::endl;
    std::cout << "Uncalibrated Confidence: " << std::exp(maxActivation) / expSum
              << std::endl;

    // Measure latency
    int numTests{100};
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    for (int i = 0; i < numTests; i++)
    {
        session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                    inputTensors.data(), 1, outputNames.data(),
                    outputTensors.data(), 1);
    }
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::cout << "Minimum Inference Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       begin)
                         .count() /
                     static_cast<float>(numTests)
              << " ms" << std::endl;
}
```
