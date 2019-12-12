#include "network_inference.hpp"
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace igg {

void* NetworkInference::ServeInputImageBuffer(const int kWidth, const int kHeight) {
  this->input_image_.create(kHeight, kWidth, CV_8UC4);
  return static_cast<void*>(this->input_image_.ptr());
}


void* NetworkInference::ServeSemanticClassConfidenceBuffer(const int kClassIndex, const int kWidth, const int kHeight) {
  if (this->semantic_class_confidences_.size()<=kClassIndex) {
    this->semantic_class_confidences_.resize(kClassIndex+1);
  }
  this->semantic_class_confidences_[kClassIndex].create(kHeight, kWidth, CV_32FC1);
  return static_cast<void*>(this->semantic_class_confidences_[kClassIndex].ptr());
}


void* NetworkInference::ServeStemKeypointConfidenceBuffer(const int kWidth, const int kHeight) {
  this->stem_keypoint_confidence_.create(kHeight, kWidth, CV_32FC1);
  return static_cast<void*>(this->stem_keypoint_confidence_.ptr());
}


void* NetworkInference::ServeStemOffsetXBuffer(const int kWidth, const int kHeight) {
  this->stem_offset_x_.create(kHeight, kWidth, CV_32FC1);
  return static_cast<void*>(this->stem_offset_x_.ptr());
}


void* NetworkInference::ServeStemOffsetYBuffer(const int kWidth, const int kHeight) {
  this->stem_offset_y_.create(kHeight, kWidth, CV_32FC1);
  return static_cast<void*>(this->stem_offset_y_.ptr());
}


cv::Mat NetworkInference::InputImage() {
  return this->input_image_; // copies the cv::Mat head only
}


cv::Mat NetworkInference::InputImageAsBgr() {
  if (this->input_image_.empty()) {
    throw std::runtime_error("No input image provided.");
  }
  std::vector<cv::Mat> channels;
  cv::split(this->input_image_, channels);
  std::vector<cv::Mat> bgr_channels = {channels[2], channels[1], channels[0]};
  cv::Mat input_image_bgr;
  cv::merge(bgr_channels, input_image_bgr);
  return input_image_bgr;
}


cv::Mat NetworkInference::InputImageAsRgb() {
  if (this->input_image_.empty()) {
    throw std::runtime_error("No input image provided.");
  }
  std::vector<cv::Mat> channels;
  cv::split(this->input_image_, channels);
  std::vector<cv::Mat> rgb_channels = {channels[0], channels[1], channels[2]};
  cv::Mat input_image_rgb;
  cv::merge(rgb_channels, input_image_rgb);
  return input_image_rgb;
}


cv::Mat NetworkInference::InputImageAsNir() {
  if (this->input_image_.empty()) {
    throw std::runtime_error("No input image provided.");
  }
  std::vector<cv::Mat> channels;
  cv::split(this->input_image_, channels);
  cv::Mat input_image_rgb;
  cv::merge({channels[3]}, input_image_rgb);
  return input_image_rgb;

}


cv::Mat NetworkInference::InputImageAsFalseColorBgr() {
  if (this->input_image_.empty()) {
    throw std::runtime_error("No input image provided.");
  }
  std::vector<cv::Mat> channels;
  cv::split(this->input_image_, channels);
  std::vector<cv::Mat> false_color_channels = {channels[2], channels[3], channels[0]};
  cv::Mat input_image_false_color;
  cv::merge(false_color_channels, input_image_false_color);
  return input_image_false_color;
}


cv::Mat NetworkInference::InputImageAsFalseColorRgb() {
  if (this->input_image_.empty()) {
    throw std::runtime_error("No input image provided.");
  }
  std::vector<cv::Mat> channels;
  cv::split(this->input_image_, channels);
  std::vector<cv::Mat> false_color_channels = {channels[0], channels[3], channels[2]};
  cv::Mat input_image_false_color;
  cv::merge(false_color_channels, input_image_false_color);
  return input_image_false_color;
}


cv::Mat NetworkInference::SemanticClassLabels() {
  return this->semantic_class_labels_; // copies the cv::Mat head only
}


cv::Mat NetworkInference::SemanticClassConfidence(const int kClassIndex) {
  if (kClassIndex>=this->semantic_class_confidences_.size()) {
    throw std::runtime_error("No inference for class with index: "+std::to_string(kClassIndex));
  }
  return this->semantic_class_confidences_[kClassIndex]; // copy the cv::Mat head here
}

cv::Mat NetworkInference::SemanticClassConfidenceAsUint8(const int kClassIndex) {
  cv::Mat confidence = this->SemanticClassConfidence(kClassIndex);
  cv::Mat confidence_converted;
  confidence *= 255.0;
  confidence.convertTo(confidence_converted, CV_8UC1);
  return confidence_converted;
}


cv::Mat NetworkInference::StemKeypointConfidence() {
  return this->stem_keypoint_confidence_; // copies the cv::Mat head only
}


cv::Mat NetworkInference::StemOffsetX() {
  return this->stem_offset_x_; // copies the cv::Mat head only
}


cv::Mat NetworkInference::StemOffsetY() {
  return this->stem_offset_y_; // copies the cv::Mat head only
}


void NetworkInference::SetStemPositions(const std::vector<cv::Vec2f>& kStemPositions) {
  this->stem_positions_ = kStemPositions;
}

std::vector<cv::Vec2f> NetworkInference::StemPositions() {
  return this->stem_positions_;
}


cv::Mat NetworkInference::MakePlot() {
  cv::Mat image = this->InputImageAsFalseColorBgr();
  cv::Mat gray;
  cv::cvtColor(image, gray, CV_BGR2GRAY);
  gray.convertTo(gray, CV_32FC1);
  gray /= 255.0;

  cv::Mat blue(gray.clone());
  cv::Mat green(gray.clone());
  cv::Mat red(gray.clone());

  cv::Mat weed_confidence = this->SemanticClassConfidence(1);
  cv::Mat sugar_beet_confidence = this->SemanticClassConfidence(2);

  blue += 0.5*sugar_beet_confidence;
  red += 0.5*weed_confidence;
  green += 0.5*weed_confidence;

  cv::Mat plot;
  std::vector<cv::Mat> channels = {blue, green, red};
  cv::merge(channels, plot);

  cv::Mat markers = cv::Mat::zeros(plot.rows, plot.cols, CV_32FC3);

  cv::Scalar color = cv::Scalar(1.0, 1.0, 1.0);
  int radius = 15;
  for(cv::Vec2f position: this->stem_positions_) {
    int x = position[0];
    int y = position[1];
    cv::circle(markers, cv::Point(x, y), radius, color, 1);
    cv::line(markers, cv::Point(x, y+radius), cv::Point(x, y+radius-10), color, 1);
    cv::line(markers, cv::Point(x, y-radius), cv::Point(x, y-radius+10), color, 1);
    cv::line(markers, cv::Point(x+radius, y), cv::Point(x+radius-10, y), color, 1);
    cv::line(markers, cv::Point(x-radius, y), cv::Point(x-radius+10, y), color, 1);
  }

  plot += 0.5*markers;

  // To 8 bit unsigned integer
  plot *= 255.0;
  plot.convertTo(plot, CV_8UC3);

  return plot;
}

} // namespace igg
