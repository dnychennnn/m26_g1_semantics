#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "library_crop_detection/network_inference.hpp"
#include "library_crop_detection/tensorrt_network.hpp"

int main() {
   //get some test image
  cv::Mat input_rgb = cv::imread("test_data/test_rgb.png", cv::IMREAD_UNCHANGED);
  cv::imshow("input_rgb", input_rgb);
  cv::cvtColor(input_rgb, input_rgb, cv::COLOR_BGR2RGB);
  cv::Mat input_nir = cv::imread("test_data/test_nir.png", cv::IMREAD_UNCHANGED);
  cv::imshow("input_nir", input_nir);

  // merge to 4 channels
  std::vector<cv::Mat> channels;
  cv::split(input_rgb, channels);
  channels.emplace_back(input_nir);
  cv::Mat input;
  cv::merge(channels, input);

  const igg::NetworkParameters kParameters;

  // create network instance
  igg::TensorrtNetwork network(kParameters);

  // load onnx file or use serialized engine if available
  const auto model_path = igg::Network::ModelsDir()/"hardnet56.onnx";
  std::cout << "Load model file: " << model_path << "\n";
  network.Load(model_path.string(), false);

  // check network status
  std::cout << "Network ready to infer: " << network.IsReadyToInfer() << "\n";

  // pass image
  igg::NetworkInference result;
  network.Infer(&result, input, false);

  cv::imshow("input_image", result.InputImageAsFalseColorBgr());
  cv::imshow("background_confidence", result.SemanticClassConfidence(0));
  cv::imshow("weed_confidence", result.SemanticClassConfidence(1));
  cv::imshow("sugar_beet_confidence", result.SemanticClassConfidence(2));
  cv::imshow("stem_keypoint_confidence", result.StemKeypointConfidence());
  cv::imshow("all", result.MakePlot());
  cv::waitKey(0);
}
