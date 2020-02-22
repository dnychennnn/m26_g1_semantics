#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_NETWORK_OUTPUT_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_NETWORK_OUTPUT_HPP_
/*!
 * @file network_output.hpp
 *
 * @version 0.1
 */

#include <opencv2/core.hpp>

namespace igg {

class NetworkOutput {

public:
  /*!
   * Get a place in memory to offload the network outputs.
   */
  void* ServeInputImageBuffer(const int kWidth, const int kHeight); // also holds the input image for convenience
  void* ServeSemanticClassConfidenceBuffer(const int kClassIndex, const int kWidth, const int kHeight);
  void* ServeStemKeypointConfidenceBuffer(const int kHeight, const int kWidth);
  void* ServeStemOffsetXBuffer(const int kHeight, const int kWidth);
  void* ServeStemOffsetYBuffer(const int kHeight, const int kWidth);
  void* ServeVotesBuffer(const int kHeight, const int kWidth);

  /*!
   * Get the offloaded outputs as cv::Mat.
   */
  const cv::Mat& InputImage() const;
  const cv::Mat& SemanticClassConfidence(const int kClassIndex) const;
  const cv::Mat& StemKeypointConfidence() const;
  const cv::Mat& StemOffsetX() const;
  const cv::Mat& StemOffsetY() const;

  /*!
   * Get a cv::Mat objects to place the porstprocessing outputs.
   */
  cv::Mat& ServeSemanticClassLabels(const int kHeight, const int kWidth);
  cv::Mat& ServeVotes(const int kHeight, const int kWidth);

  /*!
   * Get postprocessing outputs.
   */
  const cv::Mat& SemanticClassLabels() const;
  const cv::Mat& Votes() const; // from postprocessing

  /*!
   * Set extracted stem positions.
   *
   * Note each stem has three components x, y and confidence according to the accumulated votes.
   */
  void SetStemPositions(std::vector<cv::Vec3f>&& stem_ositions);

  /*!
   * Get extracted stem positions.
   *
   * Also see SetStemPositions.
   */
  const std::vector<cv::Vec3f>& StemPositions() const;

private:
  cv::Mat input_image_;
  std::vector<cv::Mat> semantic_class_confidences_;
  cv::Mat stem_keypoint_confidence_;
  cv::Mat stem_offset_x_;
  cv::Mat stem_offset_y_;

  cv::Mat semantic_class_labels_;
  cv::Mat votes_;

  std::vector<cv::Vec3f> stem_positions_;

};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_NETWORK_OUTPUT_HPP_
