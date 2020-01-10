/*!
 * @file crop_detection.cpp
 *
 * @version 0.1
 */

#include "ros_crop_detection/crop_detection.hpp"

#include <vector>
#include <string>

#include <cv_bridge/cv_bridge.h>

#include <boost/bind.hpp>
#include <opencv2/opencv.hpp>

#include "library_crop_detection/tensorrt_network.hpp"
#include "library_crop_detection/pytorch_network.hpp"
#include "library_crop_detection/network_output.hpp"

// Custom message wrapping around predicted stem positions
#include "ros_crop_detection/StemPositions.h"


namespace igg {

namespace fs = boost::filesystem;

CropDetection::CropDetection(ros::NodeHandle& node_handle,
                             const std::string& kRgbImageTopic,
                             const std::string& kNirImageTopic,
                             const std::string& kArchitectureName,
                             const NetworkParameters& kNetworkParameters,
                             const SemanticLabelerParameters& kSemanticLabelerParameters,
                             const StemExtractorParameters& kStemExtractorParameters):
      node_handle_{node_handle},
      rgb_image_subscriber_{this->node_handle_, kRgbImageTopic, 1},
      nir_image_subscriber_{this->node_handle_, kNirImageTopic, 1},
      time_synchronizer_{this->rgb_image_subscriber_, this->nir_image_subscriber_, 10} {

  ROS_INFO("Init crop detection node.");

  // Register callback
  this->time_synchronizer_.registerCallback(boost::bind(&CropDetection::Callback, this, _1, _2));

  // Advertise network output topic
  image_transport::ImageTransport transport(this->node_handle_);

  this->semantic_labels_publisher_ = transport.advertise("semantic_labels", 1);
  this->stem_positions_publisher_ = this->node_handle_.advertise<ros_crop_detection::StemPositions>("stem_positions", 1);

  this->input_bgr_publisher_ = transport.advertise("input_bgr", 1);
  this->input_nir_publisher_ = transport.advertise("input_nir", 1);
  this->input_false_color_publisher_ = transport.advertise("input_false_color", 1);

  this->visualization_publisher_ = transport.advertise("visualization", 1);
  this->visualization_semantics_publisher_ = transport.advertise("visualization_semantics", 1);
  this->visualization_keypoints_publisher_ = transport.advertise("visualization_keypoints", 1);
  this->visualization_votes_publisher_ = transport.advertise("visualization_votes", 1);

  try {
    ROS_INFO("Attempt to init TensorRT network.");
    this->network_ = std::make_unique<TensorrtNetwork>(kNetworkParameters, kSemanticLabelerParameters, kStemExtractorParameters);
    this->network_->Load((Network::ModelsDir()/(kArchitectureName+".onnx")).string(), false); // false as we do not enforce rebuilding the model
  } catch (const std::exception& kError) {
    ROS_WARN("Cannot init TensorRT network ('%s'). Trying Torch.", kError.what());
    try {
      this->network_ = std::make_unique<PytorchNetwork>(kNetworkParameters, kSemanticLabelerParameters, kStemExtractorParameters);
      this->network_->Load((Network::ModelsDir()/(kArchitectureName+".pt")).string(), false);
    } catch (const std::exception& kError) {
      ROS_ERROR("Cannot initialize any network ('%s'). Shutdown.", kError.what());
      ros::requestShutdown();
    }
  }
}


CropDetection::~CropDetection() {}


void CropDetection::Callback(const sensor_msgs::ImageConstPtr& kRgbImageMessage,
                             const sensor_msgs::ImageConstPtr& kNirImageMessage) {
  ROS_INFO("Crop detection callback.");

  if (!this->network_->IsReadyToInfer()) {
    ROS_WARN("Received image, but network is not loaded.");
    return;
  }

  // Get images as OpenCV matrices
  cv_bridge::CvImageConstPtr rgb_image_ptr;
  cv_bridge::CvImageConstPtr nir_image_ptr;

  try {
    rgb_image_ptr = cv_bridge::toCvShare(kRgbImageMessage, sensor_msgs::image_encodings::BGR8);
    nir_image_ptr = cv_bridge::toCvShare(kNirImageMessage, sensor_msgs::image_encodings::MONO8);
  } catch (const cv_bridge::Exception& kException) {
    ROS_ERROR("Error using cv_bridge: %s", kException.what());
    return;
  }

  // BGR to RGB
  cv::Mat rgb_image;
  cv::cvtColor(rgb_image_ptr->image, rgb_image, cv::COLOR_BGR2RGB);

  // Merge to create four channel image
  std::vector<cv::Mat> channels;
  cv::split(rgb_image, channels);
  channels.emplace_back(nir_image_ptr->image);
  cv::Mat rgb_nir_image;
  cv::merge(channels, rgb_nir_image);

  igg::NetworkOutput result;
  this->network_->Infer(result, rgb_nir_image);

  // Put network output into messages

  // Same header with timestamp as input
  std_msgs::Header header = kRgbImageMessage->header;

  sensor_msgs::ImagePtr semantic_labels = cv_bridge::CvImage(
      header, "mono8", result.SemanticClassLabels()).toImageMsg();
  this->semantic_labels_publisher_.publish(semantic_labels);

  const std::vector<cv::Vec3f>& kStemPositions = result.StemPositions();
  ros_crop_detection::StemPositions stem_positions_message;
  stem_positions_message.header = header;

  // Copy positions into message
  for(const auto kPosition: kStemPositions) {
    geometry_msgs::Point message_position;
    message_position.x = kPosition[0];
    message_position.y = kPosition[1];
    message_position.z = kPosition[2]; // confidence
    stem_positions_message.positions.emplace_back(message_position);
  }

  this->stem_positions_publisher_.publish(stem_positions_message);

  // Publish visualizations in debug mode
  #ifdef DEBUG_MODE

  if (this->input_bgr_publisher_.getNumSubscribers()>0) {
    sensor_msgs::ImagePtr input_bgr = cv_bridge::CvImage(
        header, "bgr8", this->kVisualizer_.MakeInputBgrVisualization(result)).toImageMsg();
    this->input_bgr_publisher_.publish(input_bgr);
  }

  if (this->input_nir_publisher_.getNumSubscribers()>0) {
    sensor_msgs::ImagePtr input_nir = cv_bridge::CvImage(
        header, "bgr8", this->kVisualizer_.MakeInputNirVisualization(result)).toImageMsg();
    this->input_nir_publisher_.publish(input_nir);
  }

  if (this->input_false_color_publisher_.getNumSubscribers()>0) {
    sensor_msgs::ImagePtr input_false_color = cv_bridge::CvImage(
        header, "bgr8", this->kVisualizer_.MakeInputFalseColorVisualization(result)).toImageMsg();
    this->input_false_color_publisher_.publish(input_false_color);
  }

  if (this->visualization_publisher_.getNumSubscribers()>0) {
    sensor_msgs::ImagePtr visualization = cv_bridge::CvImage(
        header, "bgr8", this->kVisualizer_.MakeVisualization(result)).toImageMsg();
    this->visualization_publisher_.publish(visualization);
  }

  if (this->visualization_semantics_publisher_.getNumSubscribers()>0) {
    sensor_msgs::ImagePtr visualization_semantics = cv_bridge::CvImage(
        header, "bgr8", this->kVisualizer_.MakeSemanticsVisualization(result)).toImageMsg();
    this->visualization_semantics_publisher_.publish(visualization_semantics);
  }

  if (this->visualization_keypoints_publisher_.getNumSubscribers()>0) {
    sensor_msgs::ImagePtr visualization_keypoints = cv_bridge::CvImage(
        header, "bgr8", this->kVisualizer_.MakeKeypointsVisualization(result)).toImageMsg();
    this->visualization_keypoints_publisher_.publish(visualization_keypoints);
  }

  if (this->visualization_votes_publisher_.getNumSubscribers()>0) {
    sensor_msgs::ImagePtr visualization_votes = cv_bridge::CvImage(
        header, "bgr8", this->kVisualizer_.MakeVotesVisualization(result)).toImageMsg();
    this->visualization_votes_publisher_.publish(visualization_votes);
  }

  #endif // DEBUG_MODE
}

} // namespace igg
