#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_SEMANTIC_LABELER_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_SEMANTIC_LABELER_HPP_
/*
 * @file semantic_labeler.hpp
 *
 * @version 0.1
 */

#include "library_crop_detection/semantic_labeler.hpp"

#include "library_crop_detection/network_output.hpp"


namespace igg {

struct SemanticLabelerParameters {
  float threshold_sugar_beet = 0.5;
  float threshold_weed = 0.5;
};


class SemanticLabeler {

public:
  SemanticLabeler(const SemanticLabelerParameters& kParameters);
  void Infer(NetworkOutput& result) const;

private:
  const float kThresholdSugarBeet_;
  const float kThresholdWeed_;

};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_SEMANTIC_LABELER_HPP_
