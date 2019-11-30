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

class OpencvStemInference {
public:
  OpencvStemInference(const int kKernelSizeVotes, const int kKernelSizePeaks,
      const float kThresholdVotes, const float kThresholdPeaks);

  void Infer(NetworkInference* inference) const;

private:
  const int kKernelSizeVotes_;
  const int kKernelSizePeaks_;
  const float kThresholdVotes_;
  const float kThresholdPeaks_;

};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_OPENCV_STEM_INFERENCE_HPP_
