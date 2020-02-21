#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_PYTORCH_NETWORK_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_PYTORCH_NETWORK_HPP_

 /*
 * @file pytorch_network.hpp
 *
 * @version 0.1
 */

#ifdef TORCH_AVAILABLE
#include <torch/script.h>
#endif // TORCH_AVAILABLE

#include "network.hpp"

namespace igg {

class PytorchNetwork: public Network {

public:

  PytorchNetwork(const NetworkParameters& kNetworkParameters,
      const SemanticLabelerParameters& kSemanticLabelerParameters,
      const StemExtractorParameters& kStemExtractorParameters);

  ~PytorchNetwork();

  /*!
   * See igg::Network::Infer.
   */
  void Infer(NetworkOutput& result, const cv::Mat& kImage) override;

  /*!
   * See igg::Network::IsReadyToInfer.
   */
  bool IsReadyToInfer() const override;

  /*!
   * Expects a path to an '.pt' file.
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
  const std::vector<float> kMean_;
  const std::vector<float> kStd_;

  const SemanticLabeler kSemanticLabeler_;
  const StemExtractor kStemExtractor_;

  const int kInputWidth_;
  const int kInputHeight_;
  const int kInputChannels_;

  #ifdef DEBUG_MODE
  // for debugging purposes
  std::string filepath_;
  #endif // DEBUG_MODE

  #ifdef TORCH_AVAILABLE
  torch::jit::script::Module module_;
  #endif // TORCH_AVAILABLE

  bool is_loaded_ = false;
};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_PYTORCH_NETWORK_HPP_
