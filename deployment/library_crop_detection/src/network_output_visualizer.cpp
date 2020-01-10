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

  cv::Mat plot = this->MakeBackground(kNetworkOutput, true);
  plot.convertTo(plot, CV_32FC3);
  plot /= 255.0;

  std::vector<cv::Mat> channels;
  cv::split(plot, channels);

  cv::Mat blue = channels[0];
  cv::Mat green = channels[1];
  cv::Mat red = channels[2];

  cv::Mat weed_confidence = kNetworkOutput.SemanticClassConfidence(1);
  cv::Mat sugar_beet_confidence = kNetworkOutput.SemanticClassConfidence(2);

  weed_confidence = this->Scale(weed_confidence, cv::INTER_NEAREST);
  sugar_beet_confidence = this->Scale(sugar_beet_confidence, cv::INTER_NEAREST);

  // sugar beet in blue
  blue += 0.5*sugar_beet_confidence;

  // weed in yellow
  red += 0.5*weed_confidence;
  green += 0.5*weed_confidence;

  channels = {blue, green, red};
  cv::merge(channels, plot);

  cv::Mat markers = cv::Mat::zeros(plot.rows, plot.cols, CV_32FC3);

  const cv::Mat& kInputImage = kNetworkOutput.InputImage();
  float scaling_y = static_cast<float>(this->kVisualizationHeight_)/static_cast<float>(kInputImage.rows);
  float scaling_x = static_cast<float>(this->kVisualizationWidth_)/static_cast<float>(kInputImage.cols);

  for(const cv::Vec3f& kPosition: kNetworkOutput.StemPositions()) {
    int x = static_cast<int>(std::round(scaling_x*kPosition[0]));
    int y = static_cast<int>(std::round(scaling_y*kPosition[1]));

    float confidence = std::max(std::min(kPosition[2], 1.0f), 0.1f);
    // alpha according to confidence
    cv::Scalar color = cv::Scalar(confidence, confidence, confidence);
    // thickness according to confidence
    int thickness = std::min(std::max(static_cast<int>(std::round(5.0*confidence)), 1), 5);

    cv::circle(markers, cv::Point(x, y), this->kMarkerRadius_, color, thickness);
    cv::line(markers, cv::Point(x, y+this->kMarkerRadius_), cv::Point(x, y+this->kMarkerRadius_-10), color, thickness);
    cv::line(markers, cv::Point(x, y-this->kMarkerRadius_), cv::Point(x, y-this->kMarkerRadius_+10), color, thickness);
    cv::line(markers, cv::Point(x+this->kMarkerRadius_, y), cv::Point(x+this->kMarkerRadius_-10, y), color, thickness);
    cv::line(markers, cv::Point(x-this->kMarkerRadius_, y), cv::Point(x-this->kMarkerRadius_+10, y), color, thickness);

      cv::putText(markers,
            std::to_string(kPosition[2]),
            cv::Point(x+this->kMarkerRadius_, y+this->kMarkerRadius_),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(1.0, 1.0, 1.0),
            1);
  }

  plot += 0.5*markers;

  // to 8 bit unsigned integer
  plot *= 255.0;
  plot.convertTo(plot, CV_8UC3);

  return plot;
}


cv::Mat NetworkOutputVisualizer::MakeSemanticsVisualization(
    const NetworkOutput& kNetworkOutput) const {

  cv::Mat semantic_labels = kNetworkOutput.SemanticClassLabels();

  if (semantic_labels.empty()) {
    throw std::runtime_error("No semantic labels provided.");
  }

  cv::Mat plot = this->MakeBackground(kNetworkOutput, true);

  semantic_labels = this->Scale(semantic_labels, cv::INTER_NEAREST);

  const cv::Scalar kWeedColor = cv::Scalar(0, 255, 255);
  cv::Mat weed_mask;
  cv::compare(semantic_labels, 1, weed_mask, cv::CMP_EQ);
  //plot.setTo(kWeedColor, weed_mask);

  // draw sugar beets in blue
  const cv::Scalar kSugarBeetColor = cv::Scalar(255, 0, 0);
  cv::Mat sugar_beet_mask;
  cv::compare(semantic_labels, 2, sugar_beet_mask, cv::CMP_EQ);
  //plot.setTo(kSugarBeetColor, sugar_beet_mask);

  std::vector<cv::Mat> channels;
  cv::split(plot, channels);

  cv::Mat blue = channels[0];
  cv::Mat green = channels[1];
  cv::Mat red = channels[2];

  // sugar beet in blue
  blue += 127*sugar_beet_mask;

  // weed in yellow
  red += 127*weed_mask;
  green += 127*weed_mask;

  channels = {blue, green, red};
  cv::merge(channels, plot);

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
      if (kStemKeypointConfidence.at<float>(y, x)>0.5) {
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
