#include "network_inference.hpp"

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


cv::Mat NetworkInference::PlotSemanticClassConfidences() {
  cv::Mat background = this->InputImageAsFalseColorBgr();

}

} // namespace igg
