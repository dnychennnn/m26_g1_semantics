#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_TENSORRT_COMMON_HPP
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_TENSORRT_COMMON_HPP

#include <stdexcept>
#include <string>

namespace igg {

class TensorrtNotAvailableException: public std::runtime_error {
public:
  TensorrtNotAvailableException(const char* kMessage,
      const std::string kFile, const int kLine):
      std::runtime_error(kMessage),
      kFile_(kFile),
      kLine_(kLine) {}

private:
  const std::string kFile_;
  const int kLine_;
};

} // namespace igg

#ifndef TENSORRT_AVAILABLE
#define ASSERT_TENSORRT_AVAILABLE throw igg::TensorrtNotAvailableException("TensorRT is not available in this build.", __FILE__, __LINE__)
#else
#define ASSERT_TENSORRT_AVAILABLE
#endif // TENSORRT_AVAILABLE

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_TENSORRT_COMMON_HPP
