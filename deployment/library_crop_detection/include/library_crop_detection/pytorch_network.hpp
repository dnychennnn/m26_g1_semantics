 /*
 * @file pytorch_network.hpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */


#include <torch/script.h>
#include "network.hpp"
#include "opencv_stem_inference.hpp"


namespace igg {


class PytorchNetwork: public Network {
public:


PytorchNetwork();

PytorchNetwork(const NetworkParameters& kParameters);

// ~PytorchNetwork();


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
 /*!
   * Loads and already buildt engine from file.
   *
   * @param kFilepath Path to file with the serialized engine. We usually use '.engine' as suffix.
   */
  //bool LoadSerialized(const std::string& kFilepath);
  torch::jit::script::Module* module = nullptr;

  int input_width_ = -1;
  int input_height_ = -1;
  int input_channels_ = -1;



  std::vector<float> mean_;
  std::vector<float> std_;

  const OpencvStemInference kStemInference_;

  };

}
