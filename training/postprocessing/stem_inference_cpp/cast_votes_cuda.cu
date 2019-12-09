/*!
 * @file cast_votes.cpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.hpp"

#define HANDLE_ERROR(error)(igg::handle_cuda_error(error, __FILE__, __LINE__ ))

namespace igg {

// Handling of CUDA errors.
// Reference: Jason Sander and Edward Kandrot, CUDA by Example. Retrieved from https://developer.nvidia.com/cuda-example.
// Adjusted to meet our code format.
static void handle_cuda_error(const cudaError_t kError, const char* kFile, const int kLine) {
  if (kError!=cudaSuccess) {
    throw std::runtime_error(std::string(cudaGetErrorString(kError))+" in "+kFile+" at line "+std::to_string(kLine)+".");
  }
}


__global__ void CastKernel(const long* kVotesXs,
                           const long* kVotesYs,
                           const float* kVotesWeights,
                           float* votes,
                           const int kBatchSize, const int kHeight, const int kWidth,
                           const float kThreshold) {
  const int kBatchIndex = blockIdx.y;
  const int kIndex = kBatchIndex*kHeight*kWidth+blockIdx.x*blockDim.x+threadIdx.x;

  // make sure we are inside the image
  if(kIndex>=kHeight*kWidth*kBatchSize) {return;}

  // make sure the vote is casted inside the image and weight is above threshold
  if (kVotesYs[kIndex]<0
      || kVotesYs[kIndex]>=kHeight
      || kVotesXs[kIndex]<0
      || kVotesXs[kIndex]>=kWidth
      || kVotesWeights[kIndex]<kThreshold) {return;}

  const int kVoteIndex = kBatchIndex*kHeight*kWidth+kVotesYs[kIndex]*kWidth+kVotesXs[kIndex];
  votes[kVoteIndex] += kVotesWeights[kIndex];
}


at::Tensor CastVotesCuda(torch::Tensor kVotesXs,
                         torch::Tensor kVotesYs,
                         torch::Tensor kVotesWeights,
                         const float kThreshold) {
  CHECK_DTYPE(kVotesXs, torch::kLong);
  CHECK_CONTIGUOUS(kVotesXs)
  CHECK_DIM(kVotesXs, 3);
  CHECK_DTYPE(kVotesYs, torch::kLong);
  CHECK_CONTIGUOUS(kVotesYs)
  CHECK_DIM(kVotesYs, 3);
  CHECK_DTYPE(kVotesWeights, torch::kFloat32);
  CHECK_CONTIGUOUS(kVotesWeights)
  CHECK_DIM(kVotesWeights, 3);
  CHECK_SIZE_MATCH(kVotesXs, kVotesYs, 1);
  CHECK_SIZE_MATCH(kVotesXs, kVotesYs, 2);
  CHECK_DEVICE_MATCH(kVotesXs, kVotesYs);
  CHECK_SIZE_MATCH(kVotesXs, kVotesWeights, 1);
  CHECK_SIZE_MATCH(kVotesXs, kVotesWeights, 2);
  CHECK_IS_CUDA(kVotesXs)
  CHECK_IS_CUDA(kVotesYs)
  CHECK_IS_CUDA(kVotesWeights)

  const int kBatchSize = kVotesXs.size(0);
  const int kHeight = kVotesXs.size(1);
  const int kWidth = kVotesXs.size(2);
  const int kSize = kHeight*kWidth;

  auto votes = torch::zeros_like(kVotesWeights);

  const int kNumThreads = 1024;
  // use a 2D grid of blocks, batch_index along y axis
  const dim3 kNumBlocks((kSize+kNumThreads-1)/kNumThreads, kBatchSize);

  CastKernel<<<kNumBlocks, kNumThreads>>>(kVotesXs.data<long>(), kVotesYs.data<long>(),
      kVotesWeights.data<float>(), votes.data<float>(), kBatchSize, kHeight, kWidth, kThreshold);

  HANDLE_ERROR(cudaDeviceSynchronize());

  return votes;
}

} // namespace igg

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("cast_votes_cuda", &igg::CastVotesCuda, "Cast votes (CUDA)");
}

