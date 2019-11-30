/*!
 * @file common.cpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */
#include "common.hpp"

namespace igg {

void CheckDim(const at::Tensor& kTensor, const int kDim, const std::string& kVariableName) {
  if (kTensor.dim()==kDim) {return;}
  throw std::invalid_argument("'"+kVariableName+"' has an invalid number of dimensions. "
      "Expected "+std::to_string(kDim)+ ", has "+std::to_string(kTensor.dim())+".");
}

void CheckSize(const at::Tensor& kTensor, const int kDim, const int kSize, const std::string& kVariableName) {
  if (kTensor.dim()>=kDim && kTensor.size(kDim)==kSize) {return;}
  int actual_size = -1;
  if (kTensor.dim()>=kDim) {actual_size = kTensor.size(kDim);}
  throw std::invalid_argument("'"+kVariableName+"' has an invalid size along dimension "+std::to_string(kDim)+". "
      "Expected "+std::to_string(kSize)+ ", has "+std::to_string(kTensor.size(actual_size))+".");
}

void CheckSizeMatch(const at::Tensor& kTensor1, const at::Tensor kTensor2, const int kDim,
    const std::string& kVariableName1, const std::string& kVariableName2) {
  if (kTensor1.dim()>=kDim && kTensor2.dim()>=kDim && kTensor1.size(kDim)==kTensor2.size(kDim)) {return;}
  throw std::invalid_argument("'"+kVariableName1+"' and '"+kVariableName2+"' "
      "do not match along dimension "+std::to_string(kDim)+".");
}

void CheckContiguous(const at::Tensor& kTensor, const std::string kVariableName) {
  if (kTensor.is_contiguous()) {return;}
  throw std::invalid_argument("Expected '"+kVariableName+"' to be contiguous.");
}

void CheckDtype(const at::Tensor& kTensor, const at::ScalarType kDtype, const std::string kVariableName) {
  if (kTensor.dtype()==kDtype) {return;}
  throw std::invalid_argument("'"+kVariableName+"' has an unexpected datatype.");
}

void CheckDeviceMatch(const at::Tensor& kTensor1, const at::Tensor kTensor2,
    const std::string& kVariableName1, const std::string& kVariableName2) {
  if (kTensor1.device()==kTensor2.device()) {return;}
  throw std::invalid_argument("'"+kVariableName1+"' and '"+kVariableName2+"' "
      "are not on the same device.");
}

void CheckIsCpu(const at::Tensor& kTensor, const std::string kVariableName) {
  if (kTensor.device().is_cpu()) {return;}
  throw std::invalid_argument("Expected '"+kVariableName+"' be on cpu.");
}

void CheckIsCuda(const at::Tensor& kTensor, const std::string kVariableName) {
  if (!kTensor.device().is_cpu()) {return;}
  throw std::invalid_argument("Expected '"+kVariableName+"' be on cuda device.");
}

} // namespace igg
