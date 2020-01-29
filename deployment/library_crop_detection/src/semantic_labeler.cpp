/*
 * @file semantic_labeler.cpp
 *
 * @version 0.1
 */

#include "library_crop_detection/semantic_labeler.hpp"

#ifdef DEBUG_MODE
#include <ros/console.h>
#include "library_crop_detection/stop_watch.hpp"
#endif // DEBUG_MODE


namespace igg {

SemanticLabeler::SemanticLabeler(const SemanticLabelerParameters& kParameters):
    kThresholdSugarBeet_{kParameters.threshold_sugar_beet},
    kThresholdWeed_{kParameters.threshold_weed} {}


void SemanticLabeler::Infer(NetworkOutput& result) const {
  #ifdef DEBUG_MODE
  // measure labeling time in debug mode
  StopWatch stop_watch;
  #endif // DEBUG_MODE

  const cv::Mat& kSugarBeetConfidence = result.SemanticClassConfidence(2);
  if (kSugarBeetConfidence.empty()) {
    throw std::runtime_error("Sugar beet confidences not available.");
  }

  const cv::Mat& kWeedConfidence = result.SemanticClassConfidence(1);
  if (kWeedConfidence.empty()) {
    throw std::runtime_error("Weed confidences not available.");
  }

  #ifdef DEBUG_MODE
  stop_watch.Start();
  #endif // DEBUG_MODE

  cv::Mat& semantic_labels = result.ServeSemanticClassLabels(
      kSugarBeetConfidence.rows, kSugarBeetConfidence.cols);
  semantic_labels.setTo(0); // all background

  // first assign weed labels
  kWeedConfidence.forEach<float>(
    [&](const float kConfidence, const int* kPosition) {
        if (kConfidence>this->kThresholdWeed_) {
          semantic_labels.at<float>(kPosition[0], kPosition[1]) = 1;
        }
    }
  );

  // then assign sugar beet labels
  kSugarBeetConfidence.forEach<float>(
    [&](const float kConfidence, const int* kPosition) {
        if (kConfidence>this->kThresholdSugarBeet_) {
          semantic_labels.at<float>(kPosition[0], kPosition[1]) = 2;
        }
    }
  );

  #ifdef DEBUG_MODE
  double labeling_time = stop_watch.ElapsedTime();
  ROS_INFO("Assign semantic labels: %f ms (%f fps)", 1000.0*labeling_time, 1.0/labeling_time);
  #endif // DEBUG_MODE
}

} // namespace igg
