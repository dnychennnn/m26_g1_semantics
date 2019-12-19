cmake_minimum_required(VERSION 2.8.3)
project(ros_crop_detection)

## Use CPP 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Debug and warnings
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
  cv_bridge
  image_transport
  message_filters
  library_crop_detection
)

## Find OpenCV
find_package(OpenCV 3 REQUIRED core highgui imgproc)

if(OpenCV_FOUND)
  message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
  #message(STATUS "OpenCV directories: ${OpenCV_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "OpenCV not found.")
endif(OpenCV_FOUND)

## Find Boost
find_package(Boost REQUIRED filesystem)
if(Boost_FOUND)
  message(STATUS "Found Boost version: ${Boost_VERSION}")
  #message(STATUS "Boost directories: ${Boost_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "Boost not found, please refer to README.md.")
endif(Boost_FOUND)

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  StemInference.msg
)

## Generate services in the 'srv' folder
#add_service_files(
  #FILES
#)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need

catkin_package(
  INCLUDE_DIRS
  include
  #LIBRARIES
  CATKIN_DEPENDS
    std_msgs
    geometry_msgs
    sensor_msgs
    message_runtime
    cv_bridge
    image_transport
    library_crop_detection
  #DEPENDS
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
message(WARNING "catkin_INCLUDE_DIRS  ${catkin_INCLUDE_DIRS}")

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

## Declare a cpp library
#add_library(${PROJECT_NAME}_core

#)

## Declare cpp executables
add_executable(crop_detection_node
  src/crop_detection_node.cpp
  src/crop_detection.cpp
)

## Specify libraries to link executable targets against
#target_link_libraries(${PROJECT_NAME}_core
  #${catkin_LIBRARIES}
  #${OpenCV_LIBS}
#)

target_link_libraries(crop_detection_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  Boost::filesystem
)

#add_dependencies(tester_node tester_node_generate_messages_cpp)

#############
## Install ##
#############

# Mark executables and/or libraries for installation
#install(
  #TARGETS image_extractor_node
  #ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  #LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  #RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)

# Mark cpp header files for installation
#install(
  #DIRECTORY include
  #DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  #FILES_MATCHING PATTERN "*.hpp"
#)

# Mark other files for installation
#install(
  #DIRECTORY doc
  #DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
#)

#############
## Testing ##
#############

#if(CATKIN_ENABLE_TESTING)
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  ### Add gtest based cpp test target and link libraries
  #catkin_add_gtest(${PROJECT_NAME}_test test/algorithm_test.cpp)
  #target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME}_core)
#endif()
