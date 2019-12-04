 /*
 * @file pytorch_network.hpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */


#include "network.hpp"

namespace igg {


class PytorchNetwork: public Network {
public:


PytorchNetwork();


/*!
   * See igg::Network::Infer.
   */
  NetworkInference Infer(const cv::Mat& kImage) const override;

  /*!
   * See igg::Network::IsReadyToInfer.
   */
  bool IsReadyToInfer() const override;

  /*!
   * Expects a path to an '.onnx' file.
   *
   * See igg::Network::Load.
   */
  bool Load(const std::string kFilepath, const bool kForceRebuild) override;

  /*!
   * See igg::Network.
   */
  //int InputWidth() const override;
  //int InputHeight() const override;
  //int InputChannels() const override;


private:
 /*!
   * Loads and already buildt engine from file.
   *
   * @param kFilepath Path to file with the serialized engine. We usually use '.engine' as suffix.
   */
  //bool LoadSerialized(const std::string& kFilepath);

  };

}
