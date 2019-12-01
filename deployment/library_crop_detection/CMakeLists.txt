cmake_minimum_required(VERSION 3.1)
project(library_crop_detection)

# Use CPP 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to build with TensorRT
set(CUDA_ENABLED TRUE)
set(TENSORRT_ENABLED TRUE)
message(STATUS "Enable build with CUDA: ${CUDA_ENABLED}")
message(STATUS "Enable build with TensorRT: ${TENSORRT_ENABLED}")

# Build type is debug by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
  #set(CMAKE_BUILD_TYPE Release)
endif()

# Use DEBUG_MODE preprocessor flag
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  add_definitions(-DDEBUG_MODE)
  message(STATUS "Build with debug option.")
endif()

# Warning and optimaization
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Output directories
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Find OpenCV
find_package(OpenCV 3 REQUIRED core highgui imgproc imgcodecs)

if(OpenCV_FOUND)
  message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
  #message(STATUS "OpenCV directories: ${OpenCV_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "OpenCV not found.")
endif(OpenCV_FOUND)

# Find Boost
find_package(Boost REQUIRED filesystem)
if(Boost_FOUND)
  message(STATUS "Found Boost version: ${Boost_VERSION}")
  #message(STATUS "Boost directories: ${Boost_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "Boost not found.")
endif(Boost_FOUND)

# Find Threads
#find_package (Threads)
#if(Threads_FOUND)
  #message(STATUS "Found Threads.")
#else()
  #message(STATUS "Threads not found.")
#endif(Threads_FOUND)

# Find TensorRT
if(TENSORRT_ENABLED)
  # Find CUDA
  find_package(CUDA)
  if(CUDA_FOUND)
    message(STATUS "Found CUDA version: ${CUDA_VERSION}")
  else()
    message(WARNING "CUDA not found.")
  endif(CUDA_FOUND)

  find_path(NVINFER_INCLUDE_DIRS NvInfer.h
            HINT $ENV{NVINFER_INCLUDE_DIRS})
  if(NVINFER_INCLUDE_DIRS)
    message(STATUS "Found NVINFER_INCLUDE_DIRS: ${NVINFER_INCLUDE_DIRS}")
  else()
    message(WARNING "NVINFER_INCLUDE_DIRS not found.")
  endif(NVINFER_INCLUDE_DIRS)

  find_library(NVINFER_LIBRARY libnvinfer.so
               HINT $ENV{NVINFER_LIBRARY_PATH})

  if(NVINFER_LIBRARY)
    message(STATUS "Found NVINFER_LIBRARY: ${NVINFER_LIBRARY}")
  else()
    message(WARNING "NVINFER_LIBRARY not found.")
  endif(NVINFER_LIBRARY)

  # Find TensorRT onnx parser
  find_path(NVONNXPARSER_INCLUDE_DIRS NvOnnxParser.h
            HINT $ENV{NVONNXPARSER_INCLUDE_DIRS})
  if(NVONNXPARSER_INCLUDE_DIRS)
    message(STATUS "Found NVONNXPARSER_INCLUDE_DIRS: ${NVINFER_INCLUDE_DIRS}")
  else()
    message(WARNING "NVONNXPARSER_INCLUDE_DIRS not found.")
  endif(NVONNXPARSER_INCLUDE_DIRS)

  find_library(NVONNXPARSER_LIBRARY libnvonnxparser.so
               HINT $ENV{NVONNXPARSER_LIBRARY_PATH})
  if(NVONNXPARSER_LIBRARY)
    message(STATUS "Found NVONNXPARSER_LIBRARY: ${NVONNXPARSER_LIBRARY}")
  else()
    message(WARNING "NVONNXPARSER_LIBRARY not found.")
  endif(NVONNXPARSER_LIBRARY)
endif(TENSORRT_ENABLED)

if(CUDA_ENABLED AND CUDA_FOUND)
  set(CUDA_AVAILABLE TRUE)
endif()

if(TENSORRT_ENABLED AND CUDA_AVAILABLE AND NVINFER_INCLUDE_DIRS AND NVINFER_LIBRARY)
  set(TENSORRT_AVAILABLE TRUE)
endif()

if(TENSORRT_AVAILABLE AND NVONNXPARSER_INCLUDE_DIRS AND NVONNXPARSER_LIBRARY)
  set(TENSORRT_ONNX_PARSER_AVAILABLE TRUE)
endif()

# Set preprocessor flag to selectively build with CUDA and/or TensorRT
if (CUDA_AVAILABLE)
  message(STATUS "Build with CUDA.")
  add_definitions(-DCUDA_AVAILABLE)

  # place for nvcc compiler flags
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
else()
  message(STATUS "Build without CUDA.")
endif(CUDA_AVAILABLE)

if(TENSORRT_AVAILABLE)
  message(STATUS "Build with TensorRT.")
  add_definitions(-DTENSORRT_AVAILABLE)
else()
  message(STATUS "Build without TensorRT.")
endif(TENSORRT_AVAILABLE)

if(TENSORRT_ONNX_PARSER_AVAILABLE)
  message(STATUS "Build with TensorRT ONNX Parser.")
  add_definitions(-DTENSORRT_ONNX_PARSER_AVAILABLE)
else()
  message(STATUS "Build without TensorRT ONNX Parser.")
endif(TENSORRT_ONNX_PARSER_AVAILABLE)

# Include
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${NVINFER_INCLUDE_DIRS}
)

if(TENSORRT_AVAILABLE)
  include_directories(${NVINFER_INCLUDE_DIRS})
endif(TENSORRT_AVAILABLE)

if(TENSORRT_ONNX_PARSER_AVAILABLE)
  include_directories(${NVONNXPARSER_INCLUDE_DIRS})
endif(TENSORRT_ONNX_PARSER_AVAILABLE)

# Build library
add_subdirectory("src")

# Build tests
enable_testing()
add_subdirectory("tests")

