#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "tensorrt_network.hpp"

int main() {
  // get some test image
  cv::Mat input_rgb = cv::imread("../test_data/test_rgb.png", cv::IMREAD_UNCHANGED);
  cv::imshow("input_rgb", input_rgb);
  cv::cvtColor(input_rgb, input_rgb, cv::COLOR_BGR2RGB);
  cv::Mat input_nir = cv::imread("../test_data/test_nir.png", cv::IMREAD_UNCHANGED);
  cv::imshow("input_nir", input_nir);

  // merge to 4 channels
  std::vector<cv::Mat> channels;
  cv::split(input_rgb, channels);
  channels.emplace_back(input_nir);
  cv::Mat input;
  cv::merge(channels, input);

  std::vector<float> mean {0.3855186419211486, 0.22667925089614235, 0.053568861512275835, 0.22021524472007756};
  std::vector<float> std {0.12405956089564812, 0.07152223154313433, 0.010760791977355064, 0.06568905699414836};

  // create out network instance
  igg::TensorrtNetwork network(mean, std);

  // load onnx file or use serialized engine if available
  const auto model_path = igg::Network::ModelsDir()/"simple_unet.onnx";
  std::cout << "Load model file: " << model_path << "\n";
  network.Load(model_path.string(), false);

  // check network status
  std::cout << "Network ready to infer: " << network.IsReadyToInfer() << "\n";

  // pass image
  igg::NetworkInference result;
  network.Infer(&result, input, false);
}
