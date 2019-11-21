#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_TENSORRT_NETWORK_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_TENSORRT_NETWORK_HPP_

/*!
 * @file tensorrt_network.hpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */

#include <exception>

#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#endif // TENSORRT_AVAILABLE

#include "network.hpp"

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
  /*!
   * Constructor.
   */
  TensorrtNetwork();

  /*!
   * See igg::Network::Infer.
   */
  NetworkInference Infer(const cv::Mat& kImage) const override;

  /*!
   * See igg::Network::IsReadyToInfer.
   */
  bool IsReadyToInfer() const override;

  /*!
   * Expects a path to an onnx file.
   *
   * See igg::Network::Load.
   */
  bool Load(const std::string kFilepath, const bool kForceRebuild) override;

  /*!
   * Loads and already buildt engine from file.
   *
   * @param kFilepath Path to file with the serialized engine.
   */
  bool LoadSerialized(const std::string kFilepath);


private:
  #ifdef TENSORRT_AVAILABLE
  TensorrtNetworkLogger logger_;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  #endif // TENSORRT_AVAILABLE
};


} // namespace igg


#ifndef TENSORRT_AVAILABLE
#define ASSERT_TENSORRT_AVAILABLE throw std::runtime_error("TensorRT is not available in this build.")
#else
#define ASSERT_TENSORRT_AVAILABLE
#endif // TENSORRT_AVAILABLE


#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_TENSORRT_NETWORK_HPP_
