#ifndef M26_G1_SEMANTICS_TRAINING_MODELS_POSTPROCESSING_MODULES_CAST_VOTEST_HPP_
#define M26_G1_SEMANTICS_TRAINING_MODELS_POSTPROCESSING_MODULES_CAST_VOTEST_HPP_
/*
 * @file cast_votes.hpp
 *
 * Reference: https://pytorch.org/tutorials/advanced/cpp_extension.html
 *
 * @author Jan Quakernack
 * @version 0.1
 */
#include <vector>

#include "common.hpp"

namespace igg {

/*!
 * Cast Hough-like votes using tensors as input.
 *
 * Each pixel casts one vote and has a weight and a position where the
 * vote should be casted. Note we use the absolute position and
 * not the offset from the pixel's position here.
 *
 * A check if the position lies within the image is performed
 * before the vote is casted.
 *
 * @param votes_xs, votes_ys Position where the vote should be casted.
 *     Shape (batch_size, height, width,). Datatype torch.long.
 * @param votes_weights A weight for each pixel.
 *     Shape (batch_size, height, width,). Datatype torch.float32.
 *
 * @return A tensor with the accumulated votes.
 *     Shape (batch_size, height, width,). Datatype torch.float32.
 */
at::Tensor CastVotes(const torch::Tensor& votes_xs,
                     torch::Tensor votes_ys,
                     torch::Tensor votes_weights,
                     const float kThreshold);

/*!
 * Variant on CPU.
 */
at::Tensor CastVotesCpu(torch::Tensor votes_xs,
                        torch::Tensor votes_ys,
                        torch::Tensor votes_weights,
                        const float kThreshold);

/*!
 * Variant on CUDA device.
 */
at::Tensor CastVotesCuda(torch::Tensor votes_xs,
                         torch::Tensor votes_ys,
                         torch::Tensor votes_weights,
                         const float kThreshold);

} // namespace igg


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("cast_votes_cpp", &igg::CastVotes, "Cast votes");
}

#endif // M26_G1_SEMANTICS_TRAINING_MODELS_POSTPROCESSING_MODULES_CAST_VOTEST_HPP_
