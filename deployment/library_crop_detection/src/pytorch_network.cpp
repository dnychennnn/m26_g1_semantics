
/*!
 * @file pytorch_network.cpp
 *
 * Reference: https://github.com/PRBonn/bonnetal/blob/master/deploy/src/segmentation/lib/src/netTensorRT.cpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */
#include "pytorch_network.hpp"
#include <iostream>
#include <memory>
#include <torch/script.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


namespace igg {

PytorchNetwork::PytorchNetwork():
  mean_{0.386, 0.227, 0.054, 0.220}, std_{0.124, 0.072, 0.0108, 0.066} {
  this->module = new torch::jit::script::Module();
}

// PytorchNetwork::PytorchNetwork(const NetworkParameters& kParameters):
//     mean_{kParameters.mean}, std_{kParameters.std}, kStemInference_{OpencvStemInference({kParameters})} {
//   this->module = new torch::jit::script::Module();
// }

namespace fs = boost::filesystem;

void PytorchNetwork::Infer(NetworkInference* result, const cv::Mat& kImage, const bool kMinimalInference) {
  
  // This whole part should depend on how the model is traced(on cuda or on cpu, refer to self.device in python training part)

  if(!this->module){
    throw std::runtime_error("No module loaded.");
  }

  // define cuda device
  // torch::Device device = torch::kCUDA;
  // set no grad
  torch::NoGradGuard no_grad;


  // resize to network input size
  cv::Mat input;

  cv::resize(kImage, input, cv::Size(this->input_width_, this->input_height_));

  const int kSemanticOutputWidth = this->input_width_;
  const int kSemanticOutputHeight = this->input_height_;
  // to float
  input.convertTo(input, CV_32F);
  // normalize and put into buffer
  // TODO read these from a config
  
  input.forEach<cv::Vec4f>(
    [&](cv::Vec4f& pixel, const int* position) {
      for(int channel_index=0; channel_index<this->input_channels_; channel_index++) {
        pixel[channel_index] = (pixel[channel_index]/255.0-this->mean_[channel_index])/this->std_[channel_index];
            }
        }
    );
  // size vector for tensor
  std::vector<int64_t> sizes = {1, this->input_channels_, this->input_height_, this->input_width_};


  torch::Tensor kImage_tensor;
  
  // transfer cv mat to tensor( also put into cuda device here)
  kImage_tensor = torch::from_blob(kImage.data, at::IntList(sizes));
  // put tensor into forward input type
  // this->module->to(device);

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(kImage_tensor);

  // forward
  auto output = this->module->forward(inputs);
  torch::Tensor semantic_output_tensor = output.toTuple()->elements()[0].toTensor();
  torch::Tensor classification_output_tensor = output.toTuple()->elements()[1].toTensor();
  torch::Tensor keypoint_output_tensor = output.toTuple()->elements()[2].toTensor();
  
  // get confidences as cv::Mat

  semantic_output_tensor = semantic_output_tensor.squeeze().permute({1, 2, 0});
  std::cout << semantic_output_tensor[0][0] ; 
  semantic_output_tensor = semantic_output_tensor.mul(255).clamp(0, 255).to(torch::kU8);
  std::cout << semantic_output_tensor[0][0] ; 
  cv::Mat semantic_output(cv::Size(kSemanticOutputWidth, kSemanticOutputHeight), CV_8UC3);
  std::memcpy(semantic_output.data, semantic_output_tensor.data_ptr(), sizeof(torch::kU8)*semantic_output_tensor.numel());
  std::vector<cv::Mat> semantic_class_confidences;
  
  cv::split(semantic_output, semantic_class_confidences);

  cv::imshow("background confidence", semantic_class_confidences[0]);
  cv::imshow("weed confidence", semantic_class_confidences[1]);
  cv::imshow("sugar beet confidence", semantic_class_confidences[2]);
  cv::waitKey(0);
}


bool PytorchNetwork::IsReadyToInfer() const{
  return true;
}

void PytorchNetwork::Load(const std::string& kFilepath, const bool kForceRebuild){

  try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      std::cout << kFilepath << std::endl;
      *this->module = torch::jit::load(kFilepath);
      }
  catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
      //  std::cerr << e.msg() << std::endl;
      }
  std::cout << "Load Model Succeed...\n";

  }

  int PytorchNetwork::InputWidth() const {return this->input_width_;}


  int PytorchNetwork::InputHeight() const {return this->input_height_;}


  int PytorchNetwork::InputChannels() const {return this->input_channels_;}
	
}
