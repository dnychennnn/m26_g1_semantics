#include "cast_votes.hpp"

#include <iostream>

namespace igg {

at::Tensor CastVotes(const torch::Tensor& kVotesXs,
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
  CHECK_DEVICE_MATCH(kVotesXs, kVotesWeights);

  if (kVotesXs.device().is_cpu()) {
    return CastVotesCpu(kVotesXs, kVotesYs, kVotesWeights, kThreshold);
  }

  return CastVotesCuda(kVotesXs, kVotesYs, kVotesWeights, kThreshold);
}


at::Tensor CastVotesCpu(torch::Tensor kVotesXs,
                        torch::Tensor kVotesYs,
                        torch::Tensor kVotesWeights,
                        const float kThreshold) {
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


at::Tensor CastVotesCuda(torch::Tensor kVotesXs,
                         torch::Tensor kVotesYs,
                         torch::Tensor kVotesWeights,
                         const float kThreshold) {

}


} // namespace igg

