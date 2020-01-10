/*!
 * @file crop_detection_node.cpp
 *
 * @version 0.1
 */

#include <string>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <ros/ros.h>

#include "library_crop_detection/network.hpp"
#include "library_crop_detection/semantic_labeler.hpp"
#include "library_crop_detection/stem_extractor.hpp"

#include "ros_crop_detection/crop_detection.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "crop_detection_node");
  ros::NodeHandle node_handle("~");

  std::string rgb_image_topic;
  std::string nir_image_topic;
  std::string architecture_name;

  igg::NetworkParameters network_parameters;
  igg::SemanticLabelerParameters semantic_labeler_parameters;
  igg::StemExtractorParameters stem_extractor_parameters;

  // Get image topic names from parameter server
  if (!node_handle.getParam("rgb_image_topic", rgb_image_topic) ||
      !node_handle.getParam("nir_image_topic", nir_image_topic) ||
      !node_handle.getParam("/architecture_name", architecture_name) ||
      !node_handle.getParam("/input_width", network_parameters.input_width) ||
      !node_handle.getParam("/input_height", network_parameters.input_height) ||
      !node_handle.getParam("/input_channels", network_parameters.input_channels) ||
      !node_handle.getParam("/mean", network_parameters.mean) ||
      !node_handle.getParam("/std", network_parameters.std),
      !node_handle.getParam("/threshold_sugar_beet", semantic_labeler_parameters.threshold_sugar_beet),
      !node_handle.getParam("/threshold_weed", semantic_labeler_parameters.threshold_weed),
      !node_handle.getParam("/stem_extraction_kernel_size_votes", stem_extractor_parameters.kernel_size_votes),
      !node_handle.getParam("/stem_extraction_kernel_size_peaks", stem_extractor_parameters.kernel_size_peaks),
      !node_handle.getParam("/stem_extraction_threshold_votes", stem_extractor_parameters.threshold_votes),
      !node_handle.getParam("/stem_extraction_threshold_peaks", stem_extractor_parameters.threshold_peaks)) {
    ROS_ERROR("Could not read parameters.");
    ros::requestShutdown();
    return 1;
  }

  igg::CropDetection crop_detection = igg::CropDetection(
      node_handle, rgb_image_topic, nir_image_topic, architecture_name,
      network_parameters, semantic_labeler_parameters, stem_extractor_parameters);

  ros::spin();

  return 0;
}
