#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_STEM_EXTRACTOR_GPU_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_STEM_EXTRACTOR_GPU_HPP_
/*
 * @file stem_extractor.hpp
 *
 * @version 0.1
 */

#include "library_crop_detection/network_output.hpp"
#include "library_crop_detection/stem_extractor.hpp"


namespace igg {

class StemExtractorGpu {

public:
  StemExtractorGpu(const StemExtractorParameters& kParameters);
  ~StemExtractorGpu();
  void Infer(void * keypoint_confidence, void * keypoint_offsets,
      const int kHeight, const int kWidth, NetworkOutput& result) const;

private:
  const float kKeypointRadius_;
  const int kKernelSizeVotes_;
  const int kKernelSizePeaks_;
  const float kThresholdVotes_;
  const float kThresholdPeaks_;
  const float kVotesNormalization_;

};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_STEM_EXTRACTOR_GPU_HPP_
