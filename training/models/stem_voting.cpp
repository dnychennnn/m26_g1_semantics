/*
 * Reference: https://pytorch.org/tutorials/advanced/cpp_extension.html
 *
 * @author Jan Quakernack
 */

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include <torch/extension.h>

# define CHECK_DIM(tensor, dim) igg::CheckDim(tensor, dim, #tensor);
# define CHECK_SIZE(tensor, dim, size) igg::CheckSize(tensor, dim, size, #tensor);
# define CHECK_SIZE_MATCH(tensor1, tensor2, dim) igg::CheckSizeMatch(tensor1, tensor2, dim, #tensor1, #tensor2);
# define CHECK_CONTIGUOUS(tensor) igg::CheckContiguous(tensor, #tensor);
# define CHECK_DTYPE(tensor, dtype) igg::CheckDtype(tensor, dtype, #tensor);

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
  if(kTensor.is_contiguous()) {return;}
  throw std::invalid_argument("Expected '"+kVariableName+"' to be contiguous.");
}

void CheckDtype(const at::Tensor& kTensor, const at::ScalarType kDtype, const std::string kVariableName) {
  if(kTensor.dtype()==kDtype) {return;}
  throw std::invalid_argument("'"+kVariableName+"' has an unexpected datatype.");
}

} // namespace stem_voting


std::vector<at::Tensor> CastVotes(
    torch::Tensor votes_xs,
    torch::Tensor votes_ys,
    torch::Tensor votes_weights) {

  CHECK_DTYPE(votes_xs, torch::kLong);
  CHECK_CONTIGUOUS(votes_xs)
  CHECK_DIM(votes_xs, 3);
  CHECK_DTYPE(votes_ys, torch::kLong);
  CHECK_CONTIGUOUS(votes_ys)
  CHECK_DIM(votes_ys, 3);
  CHECK_DTYPE(votes_weights, torch::kFloat32);
  CHECK_CONTIGUOUS(votes_weights)
  CHECK_DIM(votes_weights, 3);
  CHECK_SIZE_MATCH(votes_xs, votes_ys, 1);
  CHECK_SIZE_MATCH(votes_xs, votes_ys, 1);
  CHECK_SIZE_MATCH(votes_xs, votes_weights, 1);
  CHECK_SIZE_MATCH(votes_xs, votes_weights, 1);

  const int kBatchSize = votes_xs.size(0);
  const int kHeight = votes_xs.size(1);
  const int kWidth = votes_ys.size(2);

  auto votes = torch::zeros_like(votes_weights);

  for(int batch_index=0; batch_index<kBatchSize; batch_index++){
    //for(int y=0; )

  }

  //std::cout << stem_keypoint_output.size(0) << "\n";
  //std::cout << stem_keypoint_output.size(1) << "\n";
  //std::cout << stem_keypoint_output.size(2) << "\n";
  //std::cout << stem_keypoint_output.size(3) << "\n";
  //std::cout << stem_keypoint_output.dim() << "\n";

  //cv::Mat  = cv::Mat::zeros(kHeight, kWidth, CV_32F);
  //std::memcpy(tensor.data_ptr(), cv_mat.data, sizeof(float)*tensor.numel());

  return {}; // implicit std::vector
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("cast_votes", &CastVotes, "Cast votes");
}
