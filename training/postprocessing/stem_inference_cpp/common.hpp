#ifndef M26_G1_SEMANTICS_TRAINING_MODELS_STEM_INFERENCE_COMMON_HPP_
#define M26_G1_SEMANTICS_TRAINING_MODELS_STEM_INFERENCE_COMMON_HPP_
/*!
 * @file common.hpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */
#include <string>
#include <exception>
#include <stdexcept>

#include <torch/extension.h>

# define CHECK_DIM(tensor, dim) igg::CheckDim(tensor, dim, #tensor);
# define CHECK_SIZE(tensor, dim, size) igg::CheckSize(tensor, dim, size, #tensor);
# define CHECK_SIZE_MATCH(tensor1, tensor2, dim) igg::CheckSizeMatch(tensor1, tensor2, dim, #tensor1, #tensor2);
# define CHECK_CONTIGUOUS(tensor) igg::CheckContiguous(tensor, #tensor);
# define CHECK_DTYPE(tensor, dtype) igg::CheckDtype(tensor, dtype, #tensor);
# define CHECK_DEVICE_MATCH(tensor1, tensor2) igg::CheckDeviceMatch(tensor1, tensor2, #tensor1, #tensor2);
# define CHECK_IS_CPU(tensor) igg::CheckIsCpu(tensor, #tensor);
# define CHECK_IS_CUDA(tensor) igg::CheckIsCuda(tensor, #tensor);

namespace igg {

/*!
 * Helper functions to check if tensors have certain properties.
 * All throw a std::invalid_argument exception if the check is negative.
 */
void CheckDim(const at::Tensor& kTensor, const int kDim, const std::string& kVariableName);
void CheckSize(const at::Tensor& kTensor, const int kDim, const int kSize, const std::string& kVariableName);
void CheckSizeMatch(const at::Tensor& kTensor1, const at::Tensor kTensor2, const int kDim,
    const std::string& kVariableName1, const std::string& kVariableName2);
void CheckContiguous(const at::Tensor& kTensor, const std::string kVariableName);
void CheckDtype(const at::Tensor& kTensor, const at::ScalarType kDtype, const std::string kVariableName);
void CheckDeviceMatch(const at::Tensor& kTensor1, const at::Tensor kTensor2,
    const std::string& kVariableName1, const std::string& kVariableName2);
void CheckIsCpu(const at::Tensor& kTensor, const std::string kVariableName);
void CheckIsCuda(const at::Tensor& kTensor, const std::string kVariableName);

} // namespace igg

#endif // M26_G1_SEMANTICS_TRAINING_MODELS_STEM_INFERENCE_COMMON_HPP_
