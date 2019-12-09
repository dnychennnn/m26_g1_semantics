#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "pytorch_network.hpp"


int main(){


igg::PytorchNetwork network;

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

//cv::waitKey(0);

network.Load("../model.pt", false);

network.Infer(input);
}
