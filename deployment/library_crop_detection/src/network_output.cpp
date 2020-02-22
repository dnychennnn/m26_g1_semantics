/*!
 * @file network_output.hpp
 *
 * @version 0.1
 */

#include <iostream>

#include "library_crop_detection/network_output.hpp"


namespace igg {

void* NetworkOutput::ServeInputImageBuffer(const int kWidth, const int kHeight) {
  this->input_image_.create(kHeight, kWidth, CV_8UC4);
  return static_cast<void*>(this->input_image_.ptr());
}


void* NetworkOutput::ServeSemanticClassConfidenceBuffer(const int kClassIndex, const int kWidth, const int kHeight) {
  if (this->semantic_class_confidences_.size()<=kClassIndex) {
    this->semantic_class_confidences_.resize(kClassIndex+1);
  }
  this->semantic_class_confidences_[kClassIndex].create(kHeight, kWidth, CV_32FC1);
  return static_cast<void*>(this->semantic_class_confidences_[kClassIndex].ptr());
}


void* NetworkOutput::ServeStemKeypointConfidenceBuffer(const int kWidth, const int kHeight) {
  this->stem_keypoint_confidence_.create(kHeight, kWidth, CV_32FC1);
  return static_cast<void*>(this->stem_keypoint_confidence_.ptr());
}


void* NetworkOutput::ServeStemOffsetXBuffer(const int kWidth, const int kHeight) {
  this->stem_offset_x_.create(kHeight, kWidth, CV_32FC1);
  return static_cast<void*>(this->stem_offset_x_.ptr());
}


void* NetworkOutput::ServeStemOffsetYBuffer(const int kWidth, const int kHeight) {
  this->stem_offset_y_.create(kHeight, kWidth, CV_32FC1);
  return static_cast<void*>(this->stem_offset_y_.ptr());
}

void* NetworkOutput::ServeVotesBuffer(const int kHeight, const int kWidth) {
  this->votes_.create(kHeight, kWidth, CV_32FC1);
  return static_cast<void*>(this->votes_.ptr());
}


const cv::Mat& NetworkOutput::InputImage() const {
  return this->input_image_;
}


const cv::Mat& NetworkOutput::SemanticClassConfidence(const int kClassIndex) const {
  if (kClassIndex>=this->semantic_class_confidences_.size()) {
    throw std::runtime_error("No inference for class with index: "+std::to_string(kClassIndex));
  }
  return this->semantic_class_confidences_[kClassIndex];
}


const cv::Mat& NetworkOutput::StemKeypointConfidence() const {
  return this->stem_keypoint_confidence_;
}


const cv::Mat& NetworkOutput::StemOffsetX() const {
  return this->stem_offset_x_;
}


const cv::Mat& NetworkOutput::StemOffsetY() const {
  return this->stem_offset_y_;
}


cv::Mat& NetworkOutput::ServeSemanticClassLabels(const int kHeight, const int kWidth) {
  this->semantic_class_labels_.create(kHeight, kWidth, CV_32FC1);
  return this->semantic_class_labels_;
}


cv::Mat& NetworkOutput::ServeVotes(const int kHeight, const int kWidth) {
  this->votes_.create(kHeight, kWidth, CV_32FC1);
  return this->votes_;
}


const cv::Mat& NetworkOutput::SemanticClassLabels() const {
  return this->semantic_class_labels_;
}


const cv::Mat& NetworkOutput::Votes() const {
  return this->votes_;
}


void NetworkOutput::SetStemPositions(std::vector<cv::Vec3f>&& stem_positions) {
  this->stem_positions_ = stem_positions;
}


const std::vector<cv::Vec3f>& NetworkOutput::StemPositions() const {
  return this->stem_positions_;
}


} // namespace igg

