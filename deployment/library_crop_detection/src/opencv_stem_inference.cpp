#include "opencv_stem_inference.hpp"

namespace igg {

OpencvStemInference::OpencvStemInference(const int kKernelSizeVotes,
    const int kKernelSizePeaks, const float kThresholdVotes, const float kThresholdPeaks):
  kKernelSizeVotes_{kKernelSizeVotes}, kKernelSizePeaks_{kKernelSizePeaks_},
  kThresholdVotes_{kThresholdVotes_}, kThresholdPeaks_{kThresholdPeaks_} {}

void OpencvStemInference::Infer(NetworkInference* inference) const {
  //if (!inference->stem_keypoint_confidence) {
    //throw std::runtime_error("Stem keypoint confidences not available.");
  //}

  //if (!inference->stem_offset) {
    //throw std::runtime_error("Stem offset not available.");
  //}

}

} // namespace igg
