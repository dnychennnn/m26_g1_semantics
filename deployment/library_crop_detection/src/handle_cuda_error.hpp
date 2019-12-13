#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_SRC_HANDLE_CUDA_ERROR_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_SRC_HANDLE_CUDA_ERROR_HPP_

#include <exception>

#ifdef TENSORRT_AVAILABLE
// Handling of CUDA errors
// Reference: Jason Sander and Edward Kandrot, CUDA by Example. Retrieved from https://developer.nvidia.com/cuda-example.
// Adjusted to meet our code format.
static void handle_cuda_error(const cudaError_t kError, const char* kFile, const int kLine) {
  if (kError!=cudaSuccess) {
    throw std::runtime_error(std::string(cudaGetErrorString(kError))+" in "+kFile+" at line "+std::to_string(kLine)+".");
  }
}
#define HANDLE_ERROR(error)(handle_cuda_error(error, __FILE__, __LINE__ ))
#endif // TENSORRT_AVAILABLE

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_SRC_HANDLE_CUDA_ERROR_HPP_
