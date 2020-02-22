#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_TENSORRT_NETWORK_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_TENSORRT_NETWORK_HPP_
/*!
 * @file tensorrt_network.hpp
 *
 * @version 0.1
 */

#include <exception>
#include <vector>

#include "tensorrt_common.hpp"

#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#endif // TENSORRT_AVAILABLE

#include "library_crop_detection/network.hpp"
#include "library_crop_detection/semantic_labeler.hpp"
#include "library_crop_detection/stem_extractor.hpp"
#include "library_crop_detection/stem_extractor_gpu.hpp"

namespace igg {

#ifdef TENSORRT_AVAILABLE
/*!
 * Logger for notifications by TensorRT.
 */
class TensorrtNetworkLogger: public nvinfer1::ILogger {
public:
  nvinfer1::ILogger::Severity getReportableSeverity();

private:
  void log(nvinfer1::ILogger::Severity severity, const char* kMessage) override;

};
#endif // TENSORRT_AVAILABLE


class TensorrtNetwork: public Network {

public:
  TensorrtNetwork(const NetworkParameters& kNetworkParameters,
      const SemanticLabelerParameters& kSemanticLabelerParameters,
      const StemExtractorParameters& kStemExtractorParameters);

  /*!
   * Destructor.
   */
  ~TensorrtNetwork();

  /*!
   * See igg::Network::Infer.
   */
  void Infer(NetworkOutput& result, const cv::Mat& kImage) override;

  /*!
   * See igg::Network::IsReadyToInfer.
   */
  bool IsReadyToInfer() const override;

  /*!
   * Expects a path to an '.onnx' file.
   *
   * See igg::Network::Load.
   */
  void Load(const std::string& kFilepath, const bool kForceRebuild) override;

  /*!
   * See igg::Network.
   */
  int InputWidth() const override;
  int InputHeight() const override;
  int InputChannels() const override;


private:
  int input_binding_index_ = -1;
  int input_width_ = -1;
  int input_height_ = -1;
  int input_channels_ = -1;

  int semantic_output_binding_index_ = -1;
  int semantic_output_width_ = -1;
  int semantic_output_height_ = -1;
  int semantic_output_channels_ = -1;

  int stem_keypoint_output_binding_index_ = -1;
  int stem_keypoint_output_width_ = -1;
  int stem_keypoint_output_height_ = -1;
  int stem_keypoint_output_channels_ = -1;

  int stem_offset_output_binding_index_ = -1;
  int stem_offset_output_width_ = -1;
  int stem_offset_output_height_ = -1;
  int stem_offset_output_channels_ = -1;

  std::vector<float> mean_;
  std::vector<float> std_;

  void* host_buffer_ = nullptr; // for input only, memory for results is provided by igg::NetworkInference
  std::vector<void*> device_buffers_; // one for each binding

  #ifdef DEBUG_MODE
  // filepath for debug purposes
  std::string filepath_ = "";
  #endif // DEBUG_MODE

  bool engine_exists_ = false;

  #ifdef TENSORRT_AVAILABLE
  TensorrtNetworkLogger logger_;
  nvinfer1::ICudaEngine* engine_;
  #endif // TENSORRT_AVAILABLE

  const SemanticLabeler kSemanticLabeler_;

  #ifdef CUDA_AVAILABLE
  StemExtractorGpu stem_extractor_gpu_;
  #else
  const StemExtractor kStemExtractor_;
  #endif // CUDA_AVAILABLE

  /*!
   * Loads and already buildt engine from file.
   *
   * @param kFilepath Path to file with the serialized engine. We usually use '.engine' as suffix.
   */
  bool LoadSerialized(const std::string& kFilepath);

  /*!
   * Allocates memory for the engine currently loaded.
   *
   * Called by Load().
   *
   * Throws a std::runtime_error if no engine is loaded.
   */
  void ReadBindingsAndAllocateBufferMemory();

  void FreeBufferMemory();
};


} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_TENSORRT_NETWORK_HPP_
