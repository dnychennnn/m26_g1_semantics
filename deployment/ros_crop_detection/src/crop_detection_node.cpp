/*!
 * @file crop_detection_node.cpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */

#include <string>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <ros/ros.h>

#include <library_crop_detection/network_inference.hpp>
#include <library_crop_detection/tensorrt_network.hpp>

#include "crop_detection.hpp"


int main(int argc, char** argv) {
  ros::init(argc, argv, "crop_detection_node");
  ros::NodeHandle node_handle("~");

  std::string rgb_image_topic;
  std::string nir_image_topic;
  std::string path_to_model_file;

  // Get image topic names from parameter server
  if (!node_handle.getParam("rgb_image_topic", rgb_image_topic) ||
      !node_handle.getParam("nir_image_topic", nir_image_topic) ||
      !node_handle.getParam("path_to_model_file", path_to_model_file)) {
    ROS_ERROR("Could not read parameters.");
    ros::requestShutdown();
  }

  igg::CropDetection crop_detection = igg::CropDetection(
      node_handle, rgb_image_topic, nir_image_topic, path_to_model_file);

  ros::spin();

  return 0;
}
