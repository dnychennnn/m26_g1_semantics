/*!
 * @file tester_node.cpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */

#include <string>
#include <memory>

#include <ros/ros.h>

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "tester_node");
  ros::NodeHandle node_handle("~");

  nvonnxparser::IParser parser;

  std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser());

  ros::spin();
  return 0;
}
