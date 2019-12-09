#ifndef M26_G1_SEMANTICS_DEPLOYMENT_ROS_CROP_DETECTION_INCLUDE_CROP_DETECTION_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_ROS_CROP_DETECTION_INCLUDE_CROP_DETECTION_HPP_

/*!
 * @file crop_detection.hpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */

#include <memory>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_srvs/Trigger.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <library_crop_detection/network.hpp>

namespace igg {

class CropDetection {
public:
  /*!
   * Constructor.
   *
   * @param node_handle The ROS node handle.
   * @param kRgbImageTopic RGB images are expected to be published here.
   * @param kNirImageTopic NIR imagea are expected to be published here.
   */
  CropDetection(ros::NodeHandle& node_handle,
                const std::string& kRgbImageTopic,
                const std::string& kNirImageTopic,
                const std::string& kPathToModelFile);

private:
  /*!
   * @param kRgbImageMessage The received RGB image as a ROS message.
   * @param kRgbImageMessage The received NIR image as a ROS message.
   */
  void Callback(const sensor_msgs::ImageConstPtr& kRgbImageMessage,
                const sensor_msgs::ImageConstPtr& kNirImageMessage);

  //! ROS node handle.
  ros::NodeHandle& node_handle_;

  //! Subscriber to RGB image topic.
  message_filters::Subscriber<sensor_msgs::Image> rgb_image_subscriber_;
  //! Subscriber to NIR image topic.
  message_filters::Subscriber<sensor_msgs::Image> nir_image_subscriber_;

  //! Synchronizer to collect messages with the same time stamp in a single callback.
  message_filters::TimeSynchronizer <sensor_msgs::Image, sensor_msgs::Image> time_synchronizer_;

  //! Publisher for network outputs.
  image_transport::Publisher network_output_publisher_;

  //! Network.
  std::unique_ptr<igg::Network> network_;
};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_ROS_CROP_DETECTION_INCLUDE_CROP_DETECTION_HPP_
