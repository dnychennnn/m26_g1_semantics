#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_NETWORK_OUTPUT_VISUALIZER_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_NETWORK_OUTPUT_VISUALIZER_HPP_
/*!
 * @file network_output_visualizer.hpp
 *
 * @version 0.1
 */

#include <opencv2/core.hpp>

#include "library_crop_detection/network_output.hpp"

namespace igg {

class NetworkOutputVisualizer {

public:
  cv::Mat MakeVisualization(const NetworkOutput& kNetworkOutput) const;
  cv::Mat MakeSugarBeetConfidenceVisualization(const NetworkOutput& kNetworkOutput) const;
  cv::Mat MakeWeedConfidenceVisualization(const NetworkOutput& kNetworkOutput) const;
  cv::Mat MakeKeypointsVisualization(const NetworkOutput& kNetworkOutput) const;
  cv::Mat MakeVotesVisualization(const NetworkOutput& kNetworkOutput) const;

  cv::Mat MakeInputFalseColorVisualization(const NetworkOutput& kNetworkOutput) const;
  cv::Mat MakeInputBgrVisualization(const NetworkOutput& kNetworkOutput) const;
  cv::Mat MakeInputNirVisualization(const NetworkOutput& kNetworkOutput) const;

private:
  const int kVisualizationHeight_ = 644;
  const int kVisualizationWidth_ = 864;
  const int kMarkerRadius_ = 30;

  cv::Mat MakeBackground(const NetworkOutput& kNetworkOutput,
      const bool kGrayscale) const;

  cv::Mat PickInputChannels(
      const NetworkOutput& kNetworkOutput,
      const int kChannel1, const int kChannel2, const int kChannel3) const;

  cv::Mat Scale(const cv::Mat& kImage, const int kInterpolation) const;
};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_NETWORK_OUTPUT_VISUALIZER_HPP_

