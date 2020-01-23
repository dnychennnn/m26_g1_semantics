/*!
 * @file network_output_visualizer.cpp
 *
 * @version 0.1
 */

#include "library_crop_detection/network_output_visualizer.hpp"

#include <iostream>
#include <algorithm> // for min
#include <cmath> // for round

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


namespace igg {

cv::Mat NetworkOutputVisualizer::MakeVisualization(
    const NetworkOutput& kNetworkOutput) const {

  cv::Mat semantic_labels = kNetworkOutput.SemanticClassLabels();

  if (semantic_labels.empty()) {
    throw std::runtime_error("No semantic labels provided.");
  }

  cv::Mat plot = this->MakeBackground(kNetworkOutput, true);

  std::vector<cv::Mat> channels;
  cv::split(plot, channels);

  cv::Mat blue = channels[0];
  cv::Mat green = channels[1];
  cv::Mat red = channels[2];

  semantic_labels = this->Scale(semantic_labels, cv::INTER_NEAREST);

  cv::Mat weed_mask;
  cv::compare(semantic_labels, 1, weed_mask, cv::CMP_EQ);

  cv::Mat sugar_beet_mask;
  cv::compare(semantic_labels, 2, sugar_beet_mask, cv::CMP_EQ);

  // sugar beet in blue
  blue += 127*sugar_beet_mask;

  // weed in yellow
  red += 127*weed_mask;
  green += 127*weed_mask;

  channels = {blue, green, red};
  cv::merge(channels, plot);

  cv::Mat markers = cv::Mat::zeros(plot.rows, plot.cols, CV_8UC3);

  const cv::Mat& kInputImage = kNetworkOutput.InputImage();
  float scaling_y = static_cast<float>(this->kVisualizationHeight_)/static_cast<float>(kInputImage.rows);
  float scaling_x = static_cast<float>(this->kVisualizationWidth_)/static_cast<float>(kInputImage.cols);

  for(const cv::Vec3f& kPosition: kNetworkOutput.StemPositions()) {
    int x = static_cast<int>(std::round(scaling_x*kPosition[0]));
    int y = static_cast<int>(std::round(scaling_y*kPosition[1]));

    // alpha according to confidence
    int alpha = std::max(std::min(static_cast<int>(255.0*kPosition[2]), 255), 50);
    cv::Scalar color = cv::Scalar(alpha, alpha, alpha);
    // thickness according to confidence
    int thickness = std::min(std::max(static_cast<int>(std::round(5.0*alpha/255.0)), 1), 5);

    cv::circle(markers, cv::Point(x, y), this->kMarkerRadius_, color, thickness);
    cv::line(markers, cv::Point(x, y+this->kMarkerRadius_), cv::Point(x, y+this->kMarkerRadius_-10), color, thickness);
    cv::line(markers, cv::Point(x, y-this->kMarkerRadius_), cv::Point(x, y-this->kMarkerRadius_+10), color, thickness);
    cv::line(markers, cv::Point(x+this->kMarkerRadius_, y), cv::Point(x+this->kMarkerRadius_-10, y), color, thickness);
    cv::line(markers, cv::Point(x-this->kMarkerRadius_, y), cv::Point(x-this->kMarkerRadius_+10, y), color, thickness);

    cv::putText(markers, std::to_string(kPosition[2]),
        cv::Point(x+this->kMarkerRadius_, y+this->kMarkerRadius_),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  }

  plot += markers/2;

  return plot;
}


cv::Mat NetworkOutputVisualizer::MakeSugarBeetConfidenceVisualization(const NetworkOutput& kNetworkOutput) const {
  cv::Mat sugar_beet_confidence = kNetworkOutput.SemanticClassConfidence(2);

  if (sugar_beet_confidence.empty()) {
    throw std::runtime_error("No sugar beet confidence provided.");
  }

  cv::convertScaleAbs(sugar_beet_confidence, sugar_beet_confidence, 255.0);
  sugar_beet_confidence = this->Scale(sugar_beet_confidence, cv::INTER_NEAREST);

  cv::Mat plot;
  cv::applyColorMap(sugar_beet_confidence, plot, cv::COLORMAP_JET);

  return plot;
}


cv::Mat NetworkOutputVisualizer::MakeWeedConfidenceVisualization(const NetworkOutput& kNetworkOutput) const {
  cv::Mat weed_confidence = kNetworkOutput.SemanticClassConfidence(1);

  if (weed_confidence.empty()) {
    throw std::runtime_error("No weed confidence provided.");
  }

  cv::convertScaleAbs(weed_confidence, weed_confidence, 255.0);
  weed_confidence = this->Scale(weed_confidence, cv::INTER_NEAREST);

  cv::Mat plot;
  cv::applyColorMap(weed_confidence, plot, cv::COLORMAP_JET);

  return plot;
}


cv::Mat NetworkOutputVisualizer::MakeKeypointsVisualization(
    const NetworkOutput& kNetworkOutput) const {

  cv::Mat plot = this->MakeBackground(kNetworkOutput, true);

  const cv::Mat& kStemKeypointConfidence = kNetworkOutput.StemKeypointConfidence();
  if(kStemKeypointConfidence.empty()) {
    throw std::runtime_error("No stem keypoint confidences provided.");
  }

  const cv::Mat& kOffsetX = kNetworkOutput.StemOffsetX();
  if(kOffsetX.empty()) {
    throw std::runtime_error("No stem offsets x provided.");
  }

  const cv::Mat& kOffsetY = kNetworkOutput.StemOffsetY();
  if(kOffsetY.empty()) {
    throw std::runtime_error("No stem offsets y provided.");
  }

  plot.convertTo(plot, CV_32FC3);
  plot /= 255.0;

  std::vector<cv::Mat> channels;
  cv::split(plot, channels);

  cv::Mat blue = channels[0];
  cv::Mat green = channels[1];
  cv::Mat red = channels[2];

  cv::Mat stem_keypoint_confidence_scaled = this->Scale(kStemKeypointConfidence, cv::INTER_NEAREST);

  // keypoint confidence in blue
  blue = blue+0.5*stem_keypoint_confidence_scaled;

  channels = {blue, green, red};
  cv::merge(channels, plot);

  // to 8 bit unsigned integer
  plot *= 255.0;
  plot.convertTo(plot, CV_8UC3);

  float scaling_y = static_cast<float>(this->kVisualizationHeight_)/static_cast<float>(kOffsetX.rows);
  float scaling_x = static_cast<float>(this->kVisualizationWidth_)/static_cast<float>(kOffsetX.cols);

  int offset_x, offset_y;
  int scaled_x, scaled_y;
  const cv::Scalar kOffsetColor(255, 255, 255);

  for(int x=0; x<kStemKeypointConfidence.cols; x+=4) {
    for(int y=0; y<kStemKeypointConfidence.rows; y+=4) {
      if (kStemKeypointConfidence.at<float>(y, x)>0.1) {
        scaled_x = static_cast<int>(std::round(scaling_x*static_cast<float>(x)));
        scaled_y = static_cast<int>(std::round(scaling_y*static_cast<float>(y)));
        offset_x = scaled_x+static_cast<int>(std::round(15.0*scaling_x*kOffsetX.at<float>(y, x)));
        offset_y = scaled_y+static_cast<int>(std::round(15.0*scaling_y*kOffsetY.at<float>(y, x)));
        cv::arrowedLine(plot, cv::Point(scaled_x, scaled_y), cv::Point(offset_x, offset_y), kOffsetColor);
      }
    }
  }

  return plot;
}


cv::Mat NetworkOutputVisualizer::MakeVotesVisualization(
    const NetworkOutput& kNetworkOutput) const {

  cv::Mat votes = kNetworkOutput.Votes();
  if(votes.empty()) {
    throw std::runtime_error("No votes provided.");
  }

  // to 8 bit unsigned integer
  votes *= 255.0;
  votes.convertTo(votes, CV_8UC3);

  cv::applyColorMap(votes, votes, cv::COLORMAP_JET);
  votes = this->Scale(votes, cv::INTER_NEAREST);

  //cv::cvtColor(votes, votes, cv::COLOR_GRAY2BGR); // make three channel

  return votes;
}


cv::Mat NetworkOutputVisualizer::MakeInputFalseColorVisualization(
    const NetworkOutput& kNetworkOutput) const {
  cv::Mat false_color = this->PickInputChannels(kNetworkOutput, 2, 3, 0);
  return this->Scale(false_color, cv::INTER_AREA);
}


cv::Mat NetworkOutputVisualizer::MakeInputBgrVisualization(
    const NetworkOutput& kNetworkOutput) const {
  cv::Mat bgr = this->PickInputChannels(kNetworkOutput, 2, 1, 0);
  return this->Scale(bgr, cv::INTER_AREA);
}


cv::Mat NetworkOutputVisualizer::MakeInputNirVisualization(
    const NetworkOutput& kNetworkOutput) const {
  cv::Mat nir = this->PickInputChannels(kNetworkOutput, 3, 3, 3);
  return this->Scale(nir, cv::INTER_AREA);
}



cv::Mat NetworkOutputVisualizer::MakeBackground(
    const NetworkOutput& kNetworkOutput, const bool kGrayscale) const {
  cv::Mat false_color = this->MakeInputFalseColorVisualization(kNetworkOutput);

  if (kGrayscale) {
    cv::cvtColor(false_color, false_color, cv::COLOR_BGR2GRAY);
    cv::cvtColor(false_color, false_color, cv::COLOR_GRAY2BGR); // make three channel again
  }

  return this->Scale(false_color, cv::INTER_AREA);
}


cv::Mat NetworkOutputVisualizer::PickInputChannels(
    const NetworkOutput& kNetworkOutput,
    const int kChannel1, const int kChannel2, const int kChannel3) const {

  cv::Mat input_image = kNetworkOutput.InputImage();

  if (input_image.empty()) {
    throw std::runtime_error("No input image provided.");
  }

  std::vector<cv::Mat> channels;
  cv::split(input_image, channels);

  std::vector<cv::Mat> picked_channels = {channels[kChannel1], channels[kChannel2], channels[kChannel3]};
  cv::Mat picked;
  cv::merge(picked_channels, picked);

  return picked;
}



cv::Mat NetworkOutputVisualizer::Scale(const cv::Mat& kImage, const int kInterpolation) const {
  cv::Mat scaled_image;
  cv::resize(kImage, scaled_image, cv::Size(this->kVisualizationWidth_,
        this->kVisualizationHeight_), 0.0, 0.0, kInterpolation);
  return scaled_image;
}

} // namespace igg
