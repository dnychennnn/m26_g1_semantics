cmake_minimum_required(VERSION 3.1)
project(library_crop_detection)

## Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Flags to explicitly enable/disable CUDA/TensorRT
set(CUDA_ENABLED TRUE)
set(TENSORRT_ENABLED TRUE)
set(TORCH_ENABLED TRUE)
message(STATUS "Enable build with CUDA: ${CUDA_ENABLED}")
message(STATUS "Enable build with TensorRT: ${TENSORRT_ENABLED}")
message(STATUS "Enable build with Torch: ${TORCH_ENABLED}")

## Build type is debug by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
  #set(CMAKE_BUILD_TYPE Release)
endif()

## Use DEBUG_MODE preprocessor flag
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  add_definitions(-DDEBUG_MODE)
  message(STATUS "Build with debug option.")
endif()

## Warnings and optimaization
# set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED)

## Find non-ROS dependencies

## Find OpenCV
find_package(OpenCV 3 REQUIRED core highgui imgproc imgcodecs)

if(OpenCV_FOUND)
  message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
  #message(STATUS "OpenCV directories: ${OpenCV_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "OpenCV not found.")
endif(OpenCV_FOUND)

## Find Boost
find_package(Boost REQUIRED filesystem)
add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY)
if(Boost_FOUND)
  message(STATUS "Found Boost version: ${Boost_VERSION}")
  #message(STATUS "Boost directories: ${Boost_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "Boost not found.")
endif(Boost_FOUND)

## Find CUDA
if(CUDA_ENABLED OR TENSORRT_ENABLED)
  find_package(CUDA)
  if(CUDA_FOUND)
    message(STATUS "Found CUDA version: ${CUDA_VERSION}")
  else()
    message(WARNING "CUDA not found.")
  endif(CUDA_FOUND)
endif(CUDA_ENABLED OR TENSORRT_ENABLED)

## Find TensorRT
if(TENSORRT_ENABLED)
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

## Find Torch
if(TORCH_ENABLED)
  set(Torch_DIR "$ENV{LIBTORCH_CMAKE_MODULE_PATH}/Torch")
  find_package(Torch REQUIRED)
  if(TORCH_FOUND)
    message(STATUS "Found Torch.")
    #if(DEFINED ENV{TORCH_LIBRARY_PATH})
      #set(TORCH_LIBRARY_PATH $ENV{TORCH_LIBRARY_PATH})
    #else()
      #message(WARNING "Environment varibale TORCH_LIBRARY_PATH not set.")
    #endif(DEFINED ENV{TORCH_LIBRARY_PATH})
  else()
    message(WARNING "Torch not found.")
  endif(TORCH_FOUND)
endif(TORCH_ENABLED)

if(CUDA_ENABLED AND CUDA_FOUND)
  set(CUDA_AVAILABLE TRUE)
endif()

if(TENSORRT_ENABLED AND CUDA_AVAILABLE AND NVINFER_INCLUDE_DIRS AND NVINFER_LIBRARY)
  set(TENSORRT_AVAILABLE TRUE)
endif()

if(TENSORRT_AVAILABLE AND NVONNXPARSER_INCLUDE_DIRS AND NVONNXPARSER_LIBRARY)
  set(TENSORRT_ONNX_PARSER_AVAILABLE TRUE)
endif()

if(TORCH_ENABLED AND TORCH_FOUND)
  set(TORCH_AVAILABLE TRUE)
endif()

## Set preprocessor flags for CUDA specific code and nvcc flags
if (CUDA_AVAILABLE)
  message(STATUS "Build with CUDA.")
  add_definitions(-DCUDA_AVAILABLE)

  # place for nvcc compiler flags
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
else()
  message(STATUS "Build without CUDA.")
endif(CUDA_AVAILABLE)

## Set preprocessor flags for TensorRT specific code
if(TENSORRT_AVAILABLE)
  message(STATUS "Build with TensorRT.")
  add_definitions(-DTENSORRT_AVAILABLE)
else()
  message(STATUS "Build without TensorRT.")
endif(TENSORRT_AVAILABLE)

## Set preprocessor flags for TensorRT onnx parser specific code
if(TENSORRT_ONNX_PARSER_AVAILABLE)
  message(STATUS "Build with TensorRT ONNX Parser.")
  add_definitions(-DTENSORRT_ONNX_PARSER_AVAILABLE)
else()
  message(STATUS "Build without TensorRT ONNX Parser.")
endif(TENSORRT_ONNX_PARSER_AVAILABLE)

## Set preprocessor flags for Torch specific code and cmake flags for Torch
if(TORCH_AVAILABLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  add_definitions(-DTORCH_AVAILABLE)
  #list(APPEND ${TORCH_LIBRARY_PATH} CMAKE_PREFIX_PATH)
else()
  message(STATUS "Build without Torch.")
endif(TORCH_AVAILABLE)


################################################
## Declare ROS messages, services and actions ##
################################################

## No ROS messages, services ... as we only use catkin as a build tool

###################################
## catkin specific configuration ##
###################################

catkin_package(
  INCLUDE_DIRS include/${PROJECT_NAME}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  DEPENDS
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations

# Include
include_directories(
  include/${PROJECT_NAME}
  src
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

if(CUDA_AVAILABLE)
include_directories(${CUDA_INCLUDE_DIRS})
endif(CUDA_AVAILABLE)

if(TENSORRT_AVAILABLE)
  include_directories(${NVINFER_INCLUDE_DIRS})
endif(TENSORRT_AVAILABLE)

if(TENSORRT_ONNX_PARSER_AVAILABLE)
  include_directories(${NVONNXPARSER_INCLUDE_DIRS})
endif(TENSORRT_ONNX_PARSER_AVAILABLE)

if(TORCH_AVAILABLE)
  include_directories(${TORCH_INCLUDE_DIRS})
endif(TORCH_AVAILABLE)

## Declare a C++ library

add_library(${PROJECT_NAME}
  src/tensorrt_network.cpp
  src/pytorch_network.cpp
  src/network_inference.cpp
  src/opencv_stem_inference.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  Boost::filesystem
)

if(TENSORRT_AVAILABLE)
  target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${NVINFER_LIBRARY})
endif(TENSORRT_AVAILABLE)

if(TENSORRT_ONNX_PARSER_AVAILABLE)
  target_link_libraries(${PROJECT_NAME} ${NVONNXPARSER_LIBRARY})
endif(TENSORRT_ONNX_PARSER_AVAILABLE)

if(TORCH_AVAILABLE)
  target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
endif(TORCH_AVAILABLE)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/library_crop_detection_node.cpp)

add_executable(${PROJECT_NAME}_tensorrt_network_demo src/tensorrt_network_demo.cpp)
add_executable(${PROJECT_NAME}_pytorch_network_demo src/pytorch_network_demo.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
set_target_properties(${PROJECT_NAME}_tensorrt_network_demo PROPERTIES OUTPUT_NAME tensorrt_network_demo PREFIX "")
set_target_properties(${PROJECT_NAME}_pytorch_network_demo PROPERTIES OUTPUT_NAME pytorch_network_demo PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_tensorrt_network_demo
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
)

target_link_libraries(${PROJECT_NAME}_pytorch_network_demo
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
)


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

## Mark cpp header files for installation
 install(DIRECTORY include/${PROJECT_NAME}/
   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
   # FILES_MATCHING PATTERN "*.hpp"
   # PATTERN ".gitignore" EXCLUDE
 )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_library_crop_detection.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
