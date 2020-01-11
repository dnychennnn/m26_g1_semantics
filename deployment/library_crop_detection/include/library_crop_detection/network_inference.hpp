#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_NETWORK_INFERENCE_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_NETWORK_INFERENCE_HPP_
/*!
 * @file network_inference.hpp
 *
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
  void* ServeStemOffsetXBuffer(const int kHeight, const int kWidth);
  void* ServeStemOffsetYBuffer(const int kHeight, const int kWidth);
  void* ServeVotesBuffer(const int kHeight, const int kWidth);

  /*!
   * Get the offloaded outputs as cv::Mat.
   * Note that the returned cv::Mat objects still point to the data
   * stored inside this NetworkInference object.
   */
  cv::Mat InputImage();
  cv::Mat SemanticClassLabels();
  cv::Mat SemanticClassConfidence(const int kClassIndex);
  cv::Mat StemKeypointConfidence();
  cv::Mat StemOffsetX();
  cv::Mat StemOffsetY();
  cv::Mat Votes();

  /*!
   * Set accumulated votes for stem positions.
   */
  void SetVotes(const cv::Mat& votes);

  /*!
   * Set extracted stem positions.
   *
   * Note each stem has three components x, y and confidence accoring to the accumulated votes.
   */
  void SetStemPositions(const std::vector<cv::Vec2f>& kPositions);

  /*!
   * Get stem positions.
   */
  std::vector<cv::Vec2f> StemPositions();

  /*!
   * Helpers to convert to a different color mode.
   */
  cv::Mat InputImageAsRgb();
  cv::Mat InputImageAsBgr();
  cv::Mat InputImageAsNir();
  cv::Mat InputImageAsFalseColorRgb();
  cv::Mat InputImageAsFalseColorBgr();

  /*!
   * Helpers to get output as usigned 8 bit integer.
   */
  cv::Mat SemanticClassConfidenceAsUint8(const int kClassIndex);
  cv::Mat VotesAsUint8();

  /*
   * Helpers for plotting.
   */
  cv::Mat MakePlot();

private:
  cv::Mat input_image_;
  cv::Mat semantic_class_labels_;
  std::vector<cv::Mat> semantic_class_confidences_;
  cv::Mat stem_keypoint_confidence_;
  cv::Mat stem_offset_x_;
  cv::Mat stem_offset_y_;
  cv::Mat votes_;
  std::vector<cv::Vec2f> stem_positions_;
};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_NETWORK_INFERENCE_HPP_
