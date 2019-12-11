#include "pytorch_network.hpp"
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


int main(){

const igg::NetworkParameters kParameters;

// igg::PytorchNetwork network(kParameters);


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

cv::waitKey(0);

igg::NetworkInference result;

const auto model_path = igg::Network::ModelsDir()/"simple_unet.pt";
std::cout << "Load Model: " <<model_path.string() << std::endl;
network.Load(model_path.string(), false);
network.Infer(&result, input, false);
}
