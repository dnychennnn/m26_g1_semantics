## m26_g1_semantics

Semantic segmentation and stem detection for agricultural robotics using ROS.

### File structure

```
.
├── README.md
│
├── training (Python package for training of neural network with Pytorch goes here)
│
│
└── deployment (ROS package written in C++ goes here)

```

### Training phase setup

Set the following environment variables (e.g. by adding to your `.bashrc`):

```
export M26_G1_SEMANTICS_DATA_DIR="/path/to/training/dataset/"
export M26_G1_SEMANTICS_LOGS_DIR="/path/to/training/logs/"
export M26_G1_SEMANTICS_MODELS_DIR="/path/to/model/weight/files/"
export M26_G1_SEMANTICS_CONFIGS_DIR="/path/to/config/yaml/files/"
export M26_G1_SEMANTICS_CUDA_DEVICE_NAME="cuda" or "cuda:1" or ...
```

Add the python package to your python path:

```
export PYTHONPATH="${PYTHONPATH}:/path/to/your/clone/of/m26_g1_semantics/"
```

### Deployment phase setup

#### TensorRT

We currently work with CUDA 10.1 and TensorRT 6.0.1.5.

[TensorRT with installation instructions](https://github.com/NVIDIA/TensorRT)
[TensorRT backend for ONNX](https://github.com/onnx/onnx-tensorrt)

Our CMake files use hints set by environment variables to locate TensorRT related files.
If it is necessary to set these variables will propapby depend on the way TensorRT is installed on your system.

It is possible to build the code without having TensorRT installed. To explicitly build without TensorRT,
change the following line in the `CMakeLists.txt`:

```
set(TENSORRT_ENABLED TRUE)
```

To:

```
set(TENSORRT_ENABLED FALSE)
```

```
export NVINFER_INCLUDE_DIRS="/path/to/TensorRT/include/"
export NVINFER_LIBRARY_PATH="/path/to/TensorRT/lib/"

export NVONNXPARSER_INCLUDE_DIRS="/path/to/TensorRT/parsers/onnx/"
export NVONNXPARSER_LIBRARY_PATH="/path/to/TensorRT/build/out/"
```

### Coding style

**Open for discussion!**

#### C++

* [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

#### Python

* [PEP 8 Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
* [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
* [Google Style Dostrings](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
