/*!
 * @file crop_detection.cpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */

#include "ros_crop_detection/crop_detection.hpp"

#include <vector>
#include <iomanip> // For formatted output

#include <cv_bridge/cv_bridge.h>

#include <boost/bind.hpp>
#include <opencv2/opencv.hpp>

#include "library_crop_detection/tensorrt_common.hpp"

// Custom message wrapping around predicted stem positions
#include "ros_crop_detection/StemInference.h"


namespace igg {

namespace fs = boost::filesystem;

CropDetection::CropDetection(ros::NodeHandle& node_handle,
                             const std::string& kRgbImageTopic,
                             const std::string& kNirImageTopic,
                             const std::string& kPathToModelFile):
      node_handle_{node_handle},
      rgb_image_subscriber_{this->node_handle_, kRgbImageTopic, 1},
      nir_image_subscriber_{this->node_handle_, kNirImageTopic, 1},
      time_synchronizer_{this->rgb_image_subscriber_, this->nir_image_subscriber_, 10} {

  ROS_INFO("Init crop detection node.");

  // Register callback
  this->time_synchronizer_.registerCallback(boost::bind(&CropDetection::Callback, this, _1, _2));

  // Advertise network output topic
  image_transport::ImageTransport transport(this->node_handle_);

  this->network_input_image_publisher_ = transport.advertise("input_image", 1);
  this->network_background_confidence_publisher_ = transport.advertise("background_confidence", 1);
  this->network_weed_confidence_publisher_ = transport.advertise("weed_confidence", 1);
  this->network_sugar_beet_confidence_publisher_ = transport.advertise("sugar_beet_confidence", 1);
  this->network_semantic_class_labels_publisher_ = transport.advertise("semantic_class_labels", 1);
  this->network_visualization_publisher_ = transport.advertise("visualization", 1);

  this->stem_inference_publisher_ = this->node_handle_.advertise<ros_crop_detection::StemInference>("stem_inference", 1);

  try {
    ROS_INFO("Init network.");
    // this->network_ = std::make_unique<TensorrtNetwork>(NetworkParameters());
    // TODO sometimes results in a runtime error: 'free(): invalid pointer'
    this->network_.Load(kPathToModelFile+".onnx", false); // false as we do not enforce rebuilding the model
  } catch (const TensorrtNotAvailableException& kError) {
    // TODO Attempt to load pytorch model.
    ROS_ERROR("Cannot initialize TensorRT network because this is a build without TensorRT. "
              "Shutdown.");
    ros::requestShutdown();
  }
}

void CropDetection::Callback(const sensor_msgs::ImageConstPtr& kRgbImageMessage,
                             const sensor_msgs::ImageConstPtr& kNirImageMessage) {
  ROS_INFO("Crop detection callback.");

  if (!this->network_.IsReadyToInfer()) {
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

  // Show images
  //cv::imshow("RGB", rgb_image->image);
  //cv::imshow("NIR", nir_image->image);
  //cv::waitKey();

  // BGR to RGB
  cv::Mat rgb_image;
  cv::cvtColor(rgb_image_ptr->image, rgb_image, cv::COLOR_BGR2RGB);

  // Merge to create four channel image
  std::vector<cv::Mat> channels;
  cv::split(rgb_image, channels);
  channels.emplace_back(nir_image_ptr->image);
  cv::Mat rgb_nir_image;
  cv::merge(channels, rgb_nir_image);

  igg::NetworkInference result;
  this->network_.Infer(&result, rgb_nir_image, false);

  // Put network output into messages

  // Same header with timestamp as input
  std_msgs::Header header = kRgbImageMessage->header;

  sensor_msgs::ImagePtr input_image = cv_bridge::CvImage(header, "bgr8", result.InputImageAsBgr()).toImageMsg();
  sensor_msgs::ImagePtr background_confidence = cv_bridge::CvImage(header, "mono8", result.SemanticClassConfidence(0)).toImageMsg();
  sensor_msgs::ImagePtr weed_confidence = cv_bridge::CvImage(header, "", result.SemanticClassConfidence(1)).toImageMsg();
  sensor_msgs::ImagePtr sugar_beet_confidence = cv_bridge::CvImage(header, "", result.SemanticClassConfidence(2)).toImageMsg();
  sensor_msgs::ImagePtr semantic_class_labels = cv_bridge::CvImage(header, "", result.SemanticClassLabels()).toImageMsg();
  sensor_msgs::ImagePtr visualization = cv_bridge::CvImage(header, "bgr8", result.MakePlot()).toImageMsg();

  this->network_input_image_publisher_.publish(input_image);
  this->network_background_confidence_publisher_.publish(background_confidence);
  this->network_weed_confidence_publisher_.publish(weed_confidence);
  this->network_sugar_beet_confidence_publisher_.publish(sugar_beet_confidence);
  this->network_semantic_class_labels_publisher_.publish(semantic_class_labels);
  this->network_visualization_publisher_.publish(visualization);

  const std::vector<cv::Vec2f> kStemPositions = result.StemPositions();

  ros_crop_detection::StemInference stem_inference_message;
  stem_inference_message.header = header;

  // Copy positions into message
  for(const auto kPosition: kStemPositions) {
    geometry_msgs::Point message_position;
    message_position.x = kPosition[0];
    message_position.y = kPosition[1];
    message_position.z = 0.0; // geometry_msgs does not have a 2D Point. Prefer using the 3D Point over an own point message.

    stem_inference_message.positions.push_back(message_position);
  }

  this->stem_inference_publisher_.publish(stem_inference_message);

  //cv::imshow("input_image", result.InputImageAsFalseColorBgr());
  //cv::imshow("background_confidence", result.SemanticClassConfidence(0));
  //cv::imshow("weed_confidence", result.SemanticClassConfidence(1));
  //cv::imshow("sugar_beet_confidence", result.SemanticClassConfidence(2));
  //cv::imshow("stem_keypoint_confidence", result.StemKeypointConfidence());
  //cv::imshow("all", result.MakePlot());
  //cv::waitKey(1);
}

} // namespace igg
