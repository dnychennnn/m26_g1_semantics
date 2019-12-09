
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

PytorchNetwork::PytorchNetwork() {
  this->module = new torch::jit::script::Module();

}

namespace fs = boost::filesystem;

NetworkInference PytorchNetwork::Infer(const cv::Mat& kImage) const {
  
  // This whole part should depend on how the model is traced(on cuda or on cpu, refer to self.device in python training part)



  if(!this->module){
    throw std::runtime_error("No module loaded.");
  }

  // define cuda device
  torch::Device device = torch::kCUDA;


  // resize to network input size
  cv::Mat input;
  int kInputWidth = 432;
  int kInputHeight = 322;
  int kInputChannels = kImage.channels();
  cv::resize(kImage, input, cv::Size(kInputWidth, kInputHeight));
  // to float
  input.convertTo(input, CV_32F);
  // normalize and put into buffer
  // TODO read these from a config
  std::vector<float> mean {0.3855186419211486, 0.22667925089614235, 0.053568861512275835, 0.22021524472007756};
  std::vector<float> std {0.12405956089564812, 0.07152223154313433, 0.010760791977355064, 0.06568905699414836};

  input.forEach<cv::Vec4f>(
    [&](cv::Vec4f& pixel, const int* position) {
      for(int channel_index=0; channel_index<kInputChannels; channel_index++) {
        pixel[channel_index] = (pixel[channel_index]/255.0-mean[channel_index])/std[channel_index];
            }
        }
    );
  // size vector for tensor
  std::vector<int64_t> sizes = {1, kInputChannels, kInputHeight, kInputWidth};

  torch::Tensor kImage_tensor;
  try{
    // transfer cv mat to tensor( also put into cuda device here)
    kImage_tensor = torch::from_blob(kImage.data, at::IntList(sizes)).to(device);
    // put tensor into forward input type
    this->module->to(device);

  }catch(const c10::Error& e){
    // If cuda fail, use CPU 
    kImage_tensor = torch::from_blob(kImage.data, at::IntList(sizes));
  }
  
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(kImage_tensor);

  // forward
  auto output = this->module->forward(inputs).toTensor();

  std::cout << output << std::endl;
  
 
  // const int kSemanticOutputWidth;
  // const int kSemanticOutputHeight;
  // // get confidences as cv::Mat
  // cv::Mat semantic_output = cv::Mat::zeros(cv::Size(kSemanticOutputWidth, kSemanticOutputHeight), CV_32FC3);

  // semantic_output.forEach<cv::Vec3f>(
  //   [&](cv::Vec3f& pixel, const int* position) {
  //     for(int channel_index=0; channel_index<kSemanticOutputChannels; channel_index++) {
  //       const int kBufferIndex = kSemanticOutputHeight*kSemanticOutputWidth*channel_index+position[0]*kSemanticOutputWidth+position[1];
  //       pixel[channel_index] = static_cast<float*>(host_buffer[kSemanticOutputBindingIndex])[kBufferIndex];
  //     }
  //   }
  // );

  // std::vector<cv::Mat> semantic_class_confidences;
  // cv::split(semantic_output, semantic_class_confidences);

  // cv::imshow("background confidence", semantic_class_confidences[0]);
  // cv::imshow("weed confidence", semantic_class_confidences[1]);
  // cv::imshow("sugar beet confidence", semantic_class_confidences[2]);
  // cv::waitKey(0);
  
  return NetworkInference();
}


bool PytorchNetwork::Load(const std::string kFilepath, const bool kForceRebuild){

  try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      *this->module = torch::jit::load(kFilepath);
      }
  catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
      //  std::cerr << e.msg() << std::endl;
      return false;
      }
  std::cout << "ok\n";
  return true;

  }

bool PytorchNetwork::IsReadyToInfer() const{
  return true;
}

	
}
