#include <iostream>
#include <memory>

#include <torch/script.h>

#include "pytorch_network.hpp"


int main() {
  // get some test image
  cv::Mat input_rgb = cv::imread("../test_data/test_rgb.png", cv::IMREAD_UNCHANGED);
  //cv::imshow("input_rgb", input_rgb);
  cv::cvtColor(input_rgb, input_rgb, cv::COLOR_BGR2RGB);
  cv::Mat input_nir = cv::imread("../test_data/test_nir.png", cv::IMREAD_UNCHANGED);
  //cv::imshow("input_nir", input_nir);

  // merge to 4 channels
  std::vector<cv::Mat> channels;
  cv::split(input_rgb, channels);
  channels.emplace_back(input_nir);
  cv::Mat input;
  cv::merge(channels, input);

  const auto model_path = igg::Network::ModelsDir()/"simple_unet.pt";

  torch::jit::script::Module module;




  //cv::imshow("input_image", result.InputImageAsFalseColorBgr());
  //cv::imshow("background_confidence", result.SemanticClassConfidence(0));
  //cv::imshow("weed_confidence", result.SemanticClassConfidence(1));
  //cv::imshow("sugar_beet_confidence", result.SemanticClassConfidence(2));
  //cv::imshow("stem_keypoint_confidence", result.StemKeypointConfidence());
  //cv::imshow("all", result.MakePlot());
  //cv::waitKey(0);
}
