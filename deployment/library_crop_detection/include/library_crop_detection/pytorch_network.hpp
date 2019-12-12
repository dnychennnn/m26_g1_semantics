#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_PYTORCH_NETWORK_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_PYTORCH_NETWORK_HPP_

 /*
 * @file pytorch_network.hpp
 *
 * @author Yung-Yu Chen
 * @version 0.1
 */

#ifdef TORCH_AVAILABLE
#include <torch/script.h>
#endif // TORCH_AVAILABLE

#include "network.hpp"
#include "opencv_stem_inference.hpp"

namespace igg {

class PytorchNetwork: public Network {

public:

  PytorchNetwork();

  PytorchNetwork(const NetworkParameters& kParameters);

  ~PytorchNetwork();

  /*!
   * See igg::Network::Infer.
   */
  void Infer(NetworkInference* result, const cv::Mat& kImage, const bool kMinimalInference) override;

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
  #ifdef TORCH_AVAILABLE
  torch::jit::script::Module module_; // TODO maybe make this a (smart) pointer again
  #endif // TORCH_AVAILABLE

  // TODO have height, width and channels in igg::NetworkParameters and read from there
  int input_width_ = 432; //-1;
  int input_height_ = 322; //-1;
  int input_channels_ = 4; //-1;

  std::vector<float> mean_;
  std::vector<float> std_;

  void* input_buffer_ = nullptr; // for input only, memory for results is provided by igg::NetworkInference

  const OpencvStemInference kStemInference_;

};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_PYTORCH_NETWORK_HPP_
