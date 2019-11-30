/*!
 * @file cast_votes.cpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */

#include "common.hpp"

namespace igg {

at::Tensor CastVotesCpu(const torch::Tensor& kVotesXs,
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
  CHECK_IS_CPU(kVotesXs)
  CHECK_IS_CPU(kVotesYs)
  CHECK_IS_CPU(kVotesWeights)

  const int kBatchSize = kVotesXs.size(0);
  const int kHeight = kVotesXs.size(1);
  const int kWidth = kVotesYs.size(2);

  auto votes = torch::zeros_like(kVotesWeights);

  const auto kVotesXsAccesssor = kVotesXs.accessor<long, 3>();
  const auto kVotesYsAccesssor = kVotesYs.accessor<long, 3>();
  const auto kVotesWeightsAccesssor = kVotesWeights.accessor<float, 3>();
  auto votes_accessor = votes.accessor<float, 3>();

  // in principle batches could be processed in parallel also on a cpu,
  // but in evaluation as well as deployment we usually use a batch size of one
  for(int batch_index=0; batch_index<kBatchSize; batch_index++){
    for(int y=0; y<kHeight; y++){
      for(int x=0; x<kWidth; x++){
        const int kVoteY = kVotesYsAccesssor[batch_index][y][x];
        const int kVoteX = kVotesXsAccesssor[batch_index][y][x];
        const float kVoteWeight = kVotesWeightsAccesssor[batch_index][y][x];
        if (kVoteWeight>kThreshold && kVoteX>= 0 && kVoteX<kWidth && kVoteY>0 && kVoteY<kHeight) {
          votes_accessor[batch_index][kVoteY][kVoteX] += kVoteWeight;
        }
      }
    }
  }

  return votes;
}

} // namespace igg

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("cast_votes_cpu", &igg::CastVotesCpu, "Cast votes (CPU)");
}
