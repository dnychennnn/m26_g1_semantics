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
  float threshold_sugar_beet = 0.44;
  float threshold_weed = 0.99;
};


class SemanticLabeler {

public:
  SemanticLabeler(const SemanticLabelerParameters& kParameters);

  /*!
   * Assigns a semantic label based on the class confidences output by the network.
   *
   * In out implementation, this reduces to a threshold operation.
   */
  void Infer(NetworkOutput& result) const;

private:
  const float kThresholdSugarBeet_;
  const float kThresholdWeed_;

};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_SEMANTIC_LABELER_HPP_
