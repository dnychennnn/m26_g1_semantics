#include "opencv_stem_inference.hpp"

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace igg {


StemInferenceParameters::StemInferenceParameters(const NetworkParameters kParameters):
    keypoint_radius{kParameters.keypoint_radius},
    kernel_size_votes{kParameters.kernel_size_votes},
    kernel_size_peaks{kParameters.kernel_size_peaks},
    threshold_votes{kParameters.threshold_votes},
    threshold_peaks{kParameters.threshold_peaks} {}


OpencvStemInference::OpencvStemInference(const StemInferenceParameters& kParameters):
    kKeypointRadius_{kParameters.keypoint_radius},
    kKernelSizeVotes_{kParameters.kernel_size_votes},
    kKernelSizePeaks_{kParameters.kernel_size_peaks},
    kThresholdVotes_{kParameters.threshold_votes},
    kThresholdPeaks_{kParameters.threshold_peaks} {}


void OpencvStemInference::Infer(NetworkInference* inference) const {
  cv::Mat stem_keypoint_confidence = inference->StemKeypointConfidence();
  if(stem_keypoint_confidence.empty()) {
    throw std::runtime_error("Stem keypoint confidences not available.");
  }

  cv::Mat offset_x = inference->StemOffsetX();
  if(offset_x.empty()) {
    throw std::runtime_error("Stem offset x not available.");
  }

  cv::Mat offset_y = inference->StemOffsetY();
  if(offset_y.empty()) {
    throw std::runtime_error("Stem offset y not available.");
  }

  cv::Mat votes = cv::Mat::zeros(stem_keypoint_confidence.rows,
      stem_keypoint_confidence.cols, CV_32FC1);

  stem_keypoint_confidence.forEach<float>(
    [&](const float kWeight, const int* kPosition) {
        if (kWeight>this->kThresholdVotes_) {
          const int kVoteX = kPosition[1]+this->kKeypointRadius_*offset_x.at<float>(kPosition[0], kPosition[1]);
          if (kVoteX<0 || kVoteX>=votes.cols) {return;}
          const int kVoteY = kPosition[0]+this->kKeypointRadius_*offset_y.at<float>(kPosition[0], kPosition[1]);
          if (kVoteY<0 || kVoteY>=votes.rows) {return;}
          votes.at<float>(kVoteY, kVoteX) += kWeight;
        }
    }
  );

  cv::boxFilter(votes, votes, CV_32FC1, cv::Size(this->kKernelSizeVotes_, this->kKernelSizeVotes_), cv::Point(-1, -1), true, cv::BORDER_CONSTANT);

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

  std::vector<cv::Vec2f> stem_positions;

  for(int index=0; index<peak_positions.rows; index++) {
    stem_positions.emplace_back(cv::Vec2f(peak_positions.at<int>(index, 0),
                                          peak_positions.at<int>(index, 1)));
  }

  inference->SetStemPositions(std::move(stem_positions));
}

} // namespace igg
