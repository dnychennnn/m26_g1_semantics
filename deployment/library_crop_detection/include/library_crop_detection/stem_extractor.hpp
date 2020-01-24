#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_STEM_EXTRACTOR_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_STEM_EXTRACTOR_HPP_
/*
 * @file stem_extractor.hpp
 *
 * @version 0.1
 */

#include "library_crop_detection/network_output.hpp"


namespace igg {

struct StemExtractorParameters {
  float keypoint_radius = 15.0;
  int kernel_size_votes = 3;
  int kernel_size_peaks = 5;
  float threshold_votes = 0.001;
  float threshold_peaks = 0.1;
};


class StemExtractor {

public:
  StemExtractor(const StemExtractorParameters& kParameters);
  ~StemExtractor();
  void Infer(NetworkOutput& result) const;

private:
  const float kKeypointRadius_;
  const int kKernelSizeVotes_;
  const int kKernelSizePeaks_;
  const float kThresholdVotes_;
  const float kThresholdPeaks_;
  const float kVotesNormalization_;

};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_STEM_EXTRACTOR_HPP_
