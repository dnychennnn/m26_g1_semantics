/*
 * @file stem_extractor.cpp
 *
 * @version 0.1
 */

#include "library_crop_detection/stem_extractor.hpp"

#include <iostream>
#include <math.h> // for pi

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#ifdef DEBUG_MODE
#include <ros/console.h>
#include "library_crop_detection/stop_watch.hpp"
#endif // DEBUG_MODE


namespace igg {

StemExtractor::StemExtractor(const StemExtractorParameters& kParameters):
    kKeypointRadius_{kParameters.keypoint_radius},
    kKernelSizeVotes_{kParameters.kernel_size_votes},
    kKernelSizePeaks_{kParameters.kernel_size_peaks},
    kThresholdVotes_{kParameters.threshold_votes},
    kThresholdPeaks_{kParameters.threshold_peaks},
    kVotesNormalization_{static_cast<float>(M_PI)*kParameters.keypoint_radius*kParameters.keypoint_radius} {}


void StemExtractor::Infer(NetworkOutput& result) const {
  #ifdef DEBUG_MODE
  // measure extraction time in debug mode
  StopWatch stop_watch;
  #endif // DEBUG_MODE

  const cv::Mat& kStemKeypointConfidence = result.StemKeypointConfidence();
  if(kStemKeypointConfidence.empty()) {
    throw std::runtime_error("Stem keypoint confidences not available.");
  }

  const cv::Mat& kOffsetX = result.StemOffsetX();
  if(kOffsetX.empty()) {
    throw std::runtime_error("Stem offset x not available.");
  }

  const cv::Mat& kOffsetY = result.StemOffsetY();
  if(kOffsetY.empty()) {
    throw std::runtime_error("Stem offset y not available.");
  }

  #ifdef DEBUG_MODE
  stop_watch.Start();
  #endif // DEBUG_MODE

  cv::Mat& votes = result.ServeVotes(
      kStemKeypointConfidence.rows, kStemKeypointConfidence.cols);
  votes.setTo(0.0f); // fill with zeros

  kStemKeypointConfidence.forEach<float>(
    [&](const float kWeight, const int* kPosition) {
        if (kWeight>this->kThresholdVotes_) {
          const int kVoteX = kPosition[1]+this->kKeypointRadius_*kOffsetX.at<float>(kPosition[0], kPosition[1]);
          if (kVoteX<0 || kVoteX>=votes.cols) {return;}
          const int kVoteY = kPosition[0]+this->kKeypointRadius_*kOffsetY.at<float>(kPosition[0], kPosition[1]);
          if (kVoteY<0 || kVoteY>=votes.rows) {return;}
          votes.at<float>(kVoteY, kVoteX) += kWeight;
        }
    }
  );

  cv::boxFilter(votes, votes, CV_32FC1, cv::Size(this->kKernelSizeVotes_, this->kKernelSizeVotes_), cv::Point(-1, -1), false, cv::BORDER_CONSTANT);
  // false is for not normalizing the kernel

  // normalize by size of keypoint disk
  votes = votes/this->kVotesNormalization_;

  cv::Mat greater_threshold;
  cv::compare(votes, this->kThresholdPeaks_, greater_threshold, cv::CMP_GT);

  cv::Mat dilated_votes;
  cv::Mat dilation_kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(this->kKernelSizePeaks_, this->kKernelSizePeaks_), cv::Point(this->kKernelSizePeaks_/2, this->kKernelSizePeaks_/2) );
  cv::dilate(votes, dilated_votes, dilation_kernel, cv::Point(-1,-1), 1, cv::BORDER_CONSTANT, 0.0);

  cv::Mat local_maxima;
  cv::compare(votes, dilated_votes, local_maxima, cv::CMP_EQ);

  cv::Mat peaks;
  cv::bitwise_and(local_maxima, greater_threshold, peaks);

  cv::Mat peak_positions;
  cv::findNonZero(peaks, peak_positions);

  std::vector<cv::Vec3f> stem_positions;
  stem_positions.reserve(peak_positions.rows);

  // give stem positions with respect to input image
  const cv::Mat& kInputImage = result.InputImage();
  float scaling_x = static_cast<float>(kInputImage.cols)/static_cast<float>(votes.cols);
  float scaling_y = static_cast<float>(kInputImage.rows)/static_cast<float>(votes.rows);

  int position_x, position_y;
  float confidence;
  for(int index=0; index<peak_positions.rows; index++) {
    position_x = scaling_x*(0.5+peak_positions.at<int>(index, 0));
    position_y = scaling_y*(0.5+peak_positions.at<int>(index, 1));
    confidence = votes.at<float>(position_y, position_x);

    stem_positions.emplace_back(cv::Vec3f(static_cast<float>(position_x),
                                          static_cast<float>(position_y),
                                          confidence));
  }

  result.SetStemPositions(std::move(stem_positions));

  #ifdef DEBUG_MODE
  double extraction_time = stop_watch.ElapsedTime();
  ROS_INFO("Stem extraction time: %f ms (%f fps)", 1000.0*extraction_time, 1.0/extraction_time);
  #endif // DEBUG_MODE
}

} // namespace igg
