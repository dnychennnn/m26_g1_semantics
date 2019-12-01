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

namespace igg {

namespace fs = boost::filesystem;

CropDetection::CropDetection(ros::NodeHandle& node_handle,
                               const std::string& kRgbImageTopic,
                               const std::string& kNirImageTopic):
      node_handle_{node_handle},
      kRgbImageTopic_{kRgbImageTopic},
      kNirImageTopic_{kNirImageTopic},
      rgb_image_subscriber_{this->node_handle_, this->kRgbImageTopic_, 1},
      nir_image_subscriber_{this->node_handle_, this->kNirImageTopic_, 1},
      time_synchronizer_{this->rgb_image_subscriber_, this->nir_image_subscriber_, 10},
      network_{TensorrtNetwork(NetworkPrameters())} {
  this->time_synchronizer_.registerCallback(boost::bind(&CropDetection::Callback, this, _1, _2));
  ROS_INFO("Launched crop detection node.");
}

ImageExtractor::~ImageExtractor() {}

void CropDetection::Callback(const sensor_msgs::ImageConstPtr& kRgbImageMessage,
                              const sensor_msgs::ImageConstPtr& kNirImageMessage) {
  //ROS_INFO("Image extractor callback.");

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

  // Merge to create four channel image
  std::vector<cv::Mat> channels;
  cv::split(rgb_image_ptr->image, channels);
  channels.emplace_back(nir_image_ptr->image);
  cv::Mat rgb_nir_image;
  cv::merge(channels, rgb_nir_image);

  // Put images into folder with the same name as the bagfile
  const fs::path kOutputFolder = this->kOutputPath_/this->kBagfilePath_.stem();
  if (!fs::exists(kOutputFolder) &&
      !fs::create_directory(kOutputFolder)) {
    ROS_ERROR("Error creating output folder.");
    return;
  }

  // Construct filenames from name of bagfile and current index padded with zeros
  std::stringstream filename_stream;
  filename_stream << this->kBagfilePath_.stem().string();
  filename_stream << "_" << std::setw(10) << std::setfill('0') << this->image_count_;
  std::string filename_base = filename_stream.str();
  const fs::path kRgbOutputPath = kOutputFolder/(filename_base+"_RGB.png");
  const fs::path kNirOutputPath = kOutputFolder/(filename_base+"_NIR.png");
  const fs::path kRgbNirOutputPath = kOutputFolder/(filename_base+"_RGBNIR.png");

  // Write images
  ROS_INFO("Writing RGB image to %s.", kRgbOutputPath.filename().string().c_str());
  cv::imwrite(kRgbOutputPath.string(), rgb_image_ptr->image);

  ROS_INFO("Writing NIR image to %s.", kNirOutputPath.filename().string().c_str());
  cv::imwrite(kNirOutputPath.string(), nir_image_ptr->image);

  ROS_INFO("Writing RGBNIR image to %s.", kRgbNirOutputPath.filename().string().c_str());
  cv::imwrite(kRgbNirOutputPath.string(), rgb_nir_image);

  // Keep track of number of images extracted so far
  this->image_count_++;
}

} // namespace igg
