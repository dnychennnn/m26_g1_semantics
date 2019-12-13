#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_NETWORK_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_NETWORK_HPP_

/*!
 * @file network.hpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */

#include <memory>

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>

#include "network_inference.hpp"

namespace igg {

struct NetworkParameters {
  std::vector<float> mean = {0.386, 0.227, 0.054, 0.220};
  std::vector<float> std = {0.124, 0.072, 0.0108, 0.066};
  float keypoint_radius = 15.0;
  int kernel_size_votes = 5;
  int kernel_size_peaks = 9;
  float threshold_votes = 0.001;
  float threshold_peaks = 0.5;
};


class Network {
public:
  /*!
   * Perfoms plant detection on the given image.
   *
   * @param result Pointer to an instance of NetworkInference, where the output should be placed.
   * @param kImage The four channel input image as a cv::Mat. Data type is expected to be CV_8U.
   *   Images smaller or larger than the network input size will be scaled to match the desired input site.
   * @param kMinimalInference If true, only return semantic class labels and stem positions
   *   an no other intermediate results (some attributes of NetworkInference will be set to nullptr).
   *   TODO optional implement this.
   */
  virtual void Infer(NetworkInference* result, const cv::Mat& kImage, const bool kMinimalInference) = 0;

  /*!
   * @return True if the network is loaded and ready for inference, false otherwise.
   */
  virtual bool IsReadyToInfer() const = 0;

  /*!
   * Loads the network from a file, where expected filetype will depend on the used backend.
   *
   * @param kFilepath Path to file that defines the network.
   * @param kForceRebuild Make the network to parse the file again,
   *   even if an already build engine with a matching name is found.
   *   If passing this flag has an effect will depend on the type of network instantiated.
   */
  virtual void Load(const std::string& kFilepath, const bool kForceRebuild) = 0;

  /*!
   * The input dimensions may be only valid if IsReadyToInfer() and change on a call of Load().
   */
  virtual int InputWidth() const = 0;
  virtual int InputHeight() const = 0;
  virtual int InputChannels() const = 0;

  /*!
   * Get the default directory for model files set via an environment variable.
   */
  static boost::filesystem::path ModelsDir() {
    const char* kCstrModelsDir = std::getenv("M26_G1_SEMANTICS_MODELS_DIR");
    if(!kCstrModelsDir) {
      throw std::runtime_error("Environment varibale M26_G1_SEMANTICS_MODELS_DIR not set.");
    }
    return boost::filesystem::path(kCstrModelsDir);
  }

};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_NETWORK_HPP_