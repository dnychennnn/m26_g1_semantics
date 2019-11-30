#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_OPENCV_STEM_INFERENCE_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_OPENCV_STEM_INFERENCE_HPP_
/*
 * @file opencv_stem_inference.hpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */

#include "network.hpp"

namespace igg {

struct StemInferenceParameters {

  StemInferenceParameters(const NetworkParameters kParameters);

  float keypoint_radius = 15.0;
  int kernel_size_votes = 5;
  int kernel_size_peaks = 9;
  float threshold_votes = 0.005;
  float threshold_peaks = 0.5;
};


class OpencvStemInference {

public:
  OpencvStemInference(const StemInferenceParameters& kParameters);

  void Infer(NetworkInference* inference) const;

private:
  const float kKeypointRadius_;
  const int kKernelSizeVotes_;
  const int kKernelSizePeaks_;
  const float kThresholdVotes_;
  const float kThresholdPeaks_;
};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_OPENCV_STEM_INFERENCE_HPP_
