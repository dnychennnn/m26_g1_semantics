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


void* NetworkInference::ServeStemOffsetBuffer(const int kWidth, const int kHeight) {
  this->stem_offset_.create(kHeight, kWidth, CV_32FC2);
  return static_cast<void*>(this->stem_offset_.ptr());
}


cv::Mat NetworkInference::InputImage() {
  return this->input_image_; // copy the cv::Mat head here
}


cv::Mat NetworkInference::InputImageBgr() {
  std::vector<cv::Mat> channels;
  cv::split(this->input_image_, channels);
  std::vector<cv::Mat> bgr_channels = {channels[2], channels[1], channels[0]};
  cv::Mat input_image_bgr;
  cv::merge(bgr_channels, input_image_bgr);
  return input_image_bgr;
}


cv::Mat NetworkInference::InputImageRgb() {
  std::vector<cv::Mat> channels;
  cv::split(this->input_image_, channels);
  std::vector<cv::Mat> rgb_channels = {channels[0], channels[1], channels[2]};
  cv::Mat input_image_rgb;
  cv::merge(rgb_channels, input_image_rgb);
  return input_image_rgb;
}


cv::Mat NetworkInference::InputImageNir() {
  std::vector<cv::Mat> channels;
  cv::split(this->input_image_, channels);
  cv::Mat input_image_rgb;
  cv::merge({channels[3]}, input_image_rgb);
  return input_image_rgb;

}


cv::Mat NetworkInference::InputImageFalseColorBgr() {
  std::vector<cv::Mat> channels;
  cv::split(this->input_image_, channels);
  std::vector<cv::Mat> false_color_channels = {channels[2], channels[3], channels[0]};
  cv::Mat input_image_false_color;
  cv::merge(false_color_channels, input_image_false_color);
  return input_image_false_color;
}


cv::Mat NetworkInference::InputImageFalseColorRgb() {
  std::vector<cv::Mat> channels;
  cv::split(this->input_image_, channels);
  std::vector<cv::Mat> false_color_channels = {channels[0], channels[3], channels[2]};
  cv::Mat input_image_false_color;
  cv::merge(false_color_channels, input_image_false_color);
  return input_image_false_color;
}


cv::Mat NetworkInference::SemanticClassLabels() {
  return this->semantic_class_labels_; // copy the cv::Mat head here
}


cv::Mat NetworkInference::SemanticClassConfidence(const int kClassIndex) {
  if (kClassIndex>=this->semantic_class_confidences_.size()) {
    throw std::runtime_error("No inference for class with index: "+std::to_string(kClassIndex));
  }
  return this->semantic_class_confidences_[kClassIndex]; // copy the cv::Mat head here
}


cv::Mat NetworkInference::StemKeypointConfidence() {
  return this->stem_keypoint_confidence_; // copy the cv::Mat head here
}


cv::Mat NetworkInference::StemOffset() {
  return this->stem_offset_; // copy the cv::Mat head here
}

} // namespace igg
