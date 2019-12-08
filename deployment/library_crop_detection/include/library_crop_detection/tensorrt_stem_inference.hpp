#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_STEM_INFERENCE_CUDA_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_STEM_INFERENCE_CUDA_HPP_
/*
 * @file stem_inference_cuda.hpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */

#include "tensorrt_common.hpp"

#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#endif // TENSORRT_AVAILABLE

namespace igg {

class TensorrtStemInference {

};

#ifdef TENSORRT_AVAILABLE
#endif // TENSORRT_AVAILABLE

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_STEM_INFERENCE_CUDA_HPP_
