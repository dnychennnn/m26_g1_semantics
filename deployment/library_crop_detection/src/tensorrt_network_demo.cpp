#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "library_crop_detection/network_output.hpp"
#include "library_crop_detection/tensorrt_network.hpp"
#include "library_crop_detection/network_output_visualizer.hpp"

int main() {
  // get some test image
  cv::Mat input_rgb = cv::imread("test_data/test_rgb.png", cv::IMREAD_UNCHANGED);
  //cv::imshow("input_rgb", input_rgb);
  cv::cvtColor(input_rgb, input_rgb, cv::COLOR_BGR2RGB);
  cv::Mat input_nir = cv::imread("test_data/test_nir.png", cv::IMREAD_UNCHANGED);
  //cv::imshow("input_nir", input_nir);

  // merge to 4 channels
  std::vector<cv::Mat> channels;
  cv::split(input_rgb, channels);
  channels.emplace_back(input_nir);
  cv::Mat input;
  cv::merge(channels, input);

  const igg::NetworkParameters kNetworkParameters;
  const igg::SemanticLabelerParameters kSemanticLabelerParameters;
  const igg::StemExtractorParameters kStemExtractorParameters;

  // create network instance
  igg::TensorrtNetwork network(kNetworkParameters, kSemanticLabelerParameters, kStemExtractorParameters);

  // load onnx file or use serialized engine if available
  const auto model_path = igg::Network::ModelsDir()/"densenet56.onnx";
  std::cout << "Load model file: " << model_path << "\n";
  network.Load(model_path.string(), false);

  // check network status
  std::cout << "Network ready to infer: " << network.IsReadyToInfer() << "\n";

  // pass image
  igg::NetworkOutput result;
  network.Infer(result, input);

  // visualization
  igg::NetworkOutputVisualizer visualizer;

  cv::imshow("input_false_color", visualizer.MakeInputFalseColorVisualization(result));
  cv::imshow("input_bgr", visualizer.MakeInputBgrVisualization(result));
  cv::imshow("input_nir", visualizer.MakeInputNirVisualization(result));
  cv::imshow("output", visualizer.MakeVisualization(result));
  cv::imshow("output_confidence_sugar_beet", visualizer.MakeSugarBeetConfidenceVisualization(result));
  cv::imshow("output_confidence_weed", visualizer.MakeWeedConfidenceVisualization(result));
  cv::imshow("output_keypoints", visualizer.MakeKeypointsVisualization(result));
  cv::imshow("keypoint_votes", visualizer.MakeVotesVisualization(result));
  cv::waitKey(0);

  // write images to files
  //cv::imwrite("input_false_color.png", visualizer.MakeInputFalseColorVisualization(result));
  //cv::imwrite("input_bgr.png", visualizer.MakeInputBgrVisualization(result));
  //cv::imwrite("input_nir.png", visualizer.MakeInputNirVisualization(result));
  //cv::imwrite("output.png", visualizer.MakeVisualization(result));
  //cv::imwrite("output_confidence_sugar_beet", visualizer.MakeSugarBeetConfidenceVisualization(result));
  //cv::imwrite("output_confidence_weed", visualizer.MakeWeedConfidenceVisualization(result));
  //cv::imwrite("output_keypoints.png", visualizer.MakeKeypointsVisualization(result));
  //cv::imwrite("keypoint_votes.png", visualizer.MakeVotesVisualization(result));
}
