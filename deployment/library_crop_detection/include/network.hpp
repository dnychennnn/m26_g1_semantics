#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_NETWORK_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_NETWORK_HPP_

/*!
 * @file network.hpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */

#include <memory>

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>

namespace igg {

struct NetworkInference {
  cv::Mat image, semantic_class_labels, semantic_class_confidence, stem_keypoint_confidence, stem_offset;

  /*!
   * Relative to image scaled to network input dimensions.
   */
  std::vector<cv::Vec2f> stem_positions;
};


class Network {
public:
  /*!
   * Perfoms plant detection on the given image.
   *
   * @param kImage The four channel input image as a cv::Mat. Data type is expected to be CV_8U.
   *   Images smaller or larger than the network input size will be scaled to match the desired input site.
   * @return An instance of NetworkInference, which allows to access the different outcomes.
   */
  virtual NetworkInference Infer(const cv::Mat& kImage) const = 0;

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
   */
  virtual bool Load(const std::string kFilepath, const bool kForceRebuild) = 0;

  int InputWidth() const {return this->input_width_;}
  int InputHeight() const {return this->input_height_;}
  int InputChannels() const {return this->input_channels_;}

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


private:
  int input_width_ = 0;
  int input_height_ = 0;
  int input_channels_ = 0;

};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_NETWORK_HPP_
