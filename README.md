## m26_g1_semantics

Semantic segmentation and stem detection for agricultural robotics using ROS.

### Results

Trained models and evaluation results (test set) are available here:
[https://uni-bonn.sciebo.de/s/0eFLh4bsu7T9Aq4](https://uni-bonn.sciebo.de/s/0eFLh4bsu7T9Aq4)

### File structure

```
.
├── README.md
├── training (Python package for training)
└── deployment
    ├── library_crop_detection (C++ library for infernce)
    └── ros_crop_detection (ROS node wrapping around library)
```

### Training phase

#### Setup

Set the following environment variables (e.g. by adding to your `.bashrc`):

```
export M26_G1_SEMANTICS_DATA_DIR="/path/to/training/dataset/folder/"
export M26_G1_SEMANTICS_LOGS_DIR="/path/to/training/logs/folder/"
export M26_G1_SEMANTICS_MODELS_DIR="/path/to/folder/with/model/weight/files/"
export M26_G1_SEMANTICS_CONFIGS_DIR="/path/to/folder/with/config/yaml/files/"
export M26_G1_SEMANTICS_CUDA_DEVICE_NAME="cuda" or "cuda:1" or ...
```

Add the python package to your python path:

```
export PYTHONPATH="${PYTHONPATH}:/path/to/your/clone/of/m26_g1_semantics/"
```

#### Run

Training parameters are defined in `training/configs/training.yaml`. To train with these parameters, use:

```
python training/script/train.py
```

If you just want to do a quick check with some output:

```
python training/script/train.py --test-run
```

If you want to do evaluation only:

```
python training/script/train.py --only-eval
```

Not the default architecture:

```
python training/script/train.py hardnet56
```

Requires a `configs/hardnet56.yaml` to exists.

#### Freeze a model to deploy

As .onnx to load with TensorRT:

```
python training/scripts/export_model.py -t onnx
```

As .pt to load with Pytorch/Torch:

```
python training/scripts/export_model.py -t pt -d cpu
```

Note the option `-d cpu`. We tested the CPU version of Pytorch/Torch for deployment only and assume the TensorRT is used for inference on a GPU.

### Deployment phase

#### Setup

```
export M26_G1_SEMANTICS_BAGS_DIR="/path/to/folder/with/bag/files/"
export M26_G1_SEMANTICS_MODELS_DIR="/path/to/folder/with/model/weight/files/"
```

##### TensorRT

We use CUDA 10.1 and TensorRT 6.0.1.5.

* [TensorRT with installation instructions](https://github.com/NVIDIA/TensorRT)
* [TensorRT parser for ONNX](https://github.com/onnx/onnx-tensorrt)

Our CMake files use hints set by environment variables to locate TensorRT  if it is not installed system-wide:

```
export NVINFER_INCLUDE_DIRS="/path/to/TensorRT/include/"
export NVINFER_LIBRARY_PATH="/path/to/TensorRT/lib/"

export NVONNXPARSER_INCLUDE_DIRS="/path/to/TensorRT/parsers/onnx/"
export NVONNXPARSER_LIBRARY_PATH="/path/to/TensorRT/build/out/"
```

It is possible to build the code without having TensorRT installed.
To explicitly build without TensorRT,
change the following line in the `CMakeLists.txt`:

```
set(TENSORRT_ENABLED TRUE)
```

To:

```
set(TENSORRT_ENABLED FALSE)
```

##### LibTorch

A build of Libtorch can be downloaded from the [pytorch website](https://pytorch.org/).
Select `Package=Libtorch`, `Language=C++` and `CUDA=None`, which will mean to run the network
on the CPU. Add the libaries to your library path and set an environment variable which allows
our CMake script to locate Torch if it is not installed system-wide:

```
export TORCH_LIBRARY_PATH="/path/to/libtorch/libtorch/lib/"
export LD_LIBRARY_PATH=${TORCH_LIBRARY_PATH}:$LD_LIBRARY_PATH

export LIBTORCH_CMAKE_MODULE_PATH="/path/to/libtorch/share/cmake/"
```

If there are problems with the prebuild library, you may need to build from source.

### Library demo

To run a short demo:

```
cd deployment/library_crop_detection/
rosrun library_crop_detection tensorrt_network_demo
rosrun library_crop_detection pytorch_network_demo
```

Make sure the model as available in the required format (`.onnx`, `.pt`) under `M26_G1_SEMANTICS_MODELS_DIR` using the provided export script.

Note `rosrun` is only used here for convinience because catkin is used as a build system throught the project.
The library is independent of ROS (exept for a few calls of `ROS_INFO` when compiled in `DEBUG_MODE` to produce readable debug output).

### ROS demo

```
roslaunch ros_crop_detection demo.launch
```

To adjust parameters, edit `deployment/ros_crop_detection/config/demo.yaml`.

