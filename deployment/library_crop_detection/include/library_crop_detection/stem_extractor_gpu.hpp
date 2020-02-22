#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_STEM_EXTRACTOR_GPU_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_STEM_EXTRACTOR_GPU_HPP_
/*
 * @file stem_extractor_gpu.hpp
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

  void LoadAndAllocateBuffers(
      const unsigned int kHeight,
      const unsigned int kWidth,
      const float kScaling);

  template <const unsigned int kTileWidth = 32,
            const unsigned int kTileHeight = 32,
            const unsigned int kNumThreads = 1024,
            const unsigned int kNumBlocks = 288,
            const unsigned int kDeviceIndex = 0>
  void Infer(
      float* keypoint_confidence_device,
      float* keypoint_offsets_device,
      NetworkOutput& result) const;

private:
  const float kKeypointRadius_;
  const int kKernelSizeVotes_;
  const int kKernelSizePeaks_;
  const float kThresholdVotes_;
  const float kThresholdPeaks_;
  const float kVotesNormalization_;

  bool is_loaded_ = false;

  unsigned int height_ = 0;
  unsigned int width_ = 0;
  unsigned int size_ = 0;
  float scaling_ = 1.0;

  float* votes_device_;
  float* device_buffer_1_;
  float* device_buffer_2_;
  unsigned int* num_stem_positions_device_;
  unsigned int* num_stem_positions_host_;
  float* stem_positions_device_;
  float* stem_positions_host_;

  void FreeBuffers();

};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_STEM_EXTRACTOR_GPU_HPP_
