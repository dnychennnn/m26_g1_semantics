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

### Coding style

**Open for discussion!**

#### C++

* [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

#### Python

* [PEP 8 Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
* [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
* [Google Style Dostrings](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
