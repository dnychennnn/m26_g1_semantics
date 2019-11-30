#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_NETWORK_INFERENCE_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_NETWORK_INFERENCE_HPP_

/*!
 * @file network_inference.hpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */

#include <opencv2/core.hpp>

namespace igg {

class NetworkInference {

public:
  /*!
   * Get a place in memory to offload the network outputs.
   */
  void* ServeInputImageBuffer(const int kWidth, const int kHeight);
  void* ServeSemanticClassConfidenceBuffer(const int kClassIndex, const int kWidth, const int kHeight);
  void* ServeStemKeypointConfidenceBuffer(const int kHeight, const int kWidth);
  void* ServeStemOffsetBuffer(const int kHeight, const int kWidth);

  /*!
   * Get the offloaded outputs as cv::Mat.
   * Note that the returned cv::Mat objects still point to the data
   * stored inside this NetworkInference object.
   */
  cv::Mat InputImage();
  cv::Mat InputImageRgb();
  cv::Mat InputImageBgr();
  cv::Mat InputImageNir();
  cv::Mat InputImageFalseColorRgb();
  cv::Mat InputImageFalseColorBgr();
  cv::Mat SemanticClassLabels();
  cv::Mat SemanticClassConfidence(const int kClassIndex);
  cv::Mat StemKeypointConfidence();
  cv::Mat StemOffset();

private:
  cv::Mat input_image_;
  cv::Mat semantic_class_labels_;
  std::vector<cv::Mat> semantic_class_confidences_;
  cv::Mat stem_keypoint_confidence_;
  cv::Mat stem_offset_;
  std::vector<cv::Vec2f> stem_positions_;
};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_NETWORK_INFERENCE_HPP_
