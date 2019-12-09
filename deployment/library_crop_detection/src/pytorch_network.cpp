
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
#include <ATen/ATen.h>

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
