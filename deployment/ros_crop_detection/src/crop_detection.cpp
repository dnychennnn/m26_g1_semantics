/*!
 * @file crop_detection.cpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */

#include "crop_detection.hpp"

#include <vector>
#include <iomanip> // For formatted output

#include <cv_bridge/cv_bridge.h>

#include <boost/bind.hpp>
#include <opencv2/opencv.hpp>

#include <library_crop_detection/tensorrt_network.hpp>
#include <library_crop_detection/tensorrt_common.hpp>

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

  // Advertise network ouput topic
  image_transport::ImageTransport transport(this->node_handle_);
  this->network_output_publisher_ = transport.advertise("network_output", 1);

  try {
    ROS_INFO("Init network.");
    this->network_ = std::make_unique<TensorrtNetwork>(NetworkParameters());
    this->network_->Load(kPathToModelFile+".onnx", false); // false as we do not enforce rebuilding the model
    // TODO sometimes results in a runtime error: 'free(): invalid pointer'
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

  if (!this->network_->IsReadyToInfer()) {
    ROS_WARN("Received image, but network is not loaded.");
    return;
  }

  // Get images as OpenCV matrices
  cv_bridge::CvImagePtr rgb_image_ptr;
  cv_bridge::CvImagePtr nir_image_ptr;

  try {
    rgb_image_ptr = cv_bridge::toCvCopy(kRgbImageMessage, sensor_msgs::image_encodings::BGR8);
    nir_image_ptr = cv_bridge::toCvCopy(kNirImageMessage, sensor_msgs::image_encodings::MONO8);
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
  this->network_->Infer(&result, rgb_nir_image, false);

  // Put output image into message
  sensor_msgs::ImagePtr message = cv_bridge::CvImage(
      std_msgs::Header(), "bgr8", result.MakePlot()).toImageMsg();

  this->network_output_publisher_.publish(message);

  //cv::imshow("input_image", result.InputImageAsFalseColorBgr());
  //cv::imshow("background_confidence", result.SemanticClassConfidence(0));
  //cv::imshow("weed_confidence", result.SemanticClassConfidence(1));
  //cv::imshow("sugar_beet_confidence", result.SemanticClassConfidence(2));
  //cv::imshow("stem_keypoint_confidence", result.StemKeypointConfidence());
  //cv::imshow("all", result.MakePlot());
  //cv::waitKey(1);
}

} // namespace igg
