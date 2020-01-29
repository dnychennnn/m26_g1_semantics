/*!
 * @file pytorch_network.cpp
 *
 * @version 0.1
 */

#include "library_crop_detection/pytorch_network.hpp"

#include <iostream>
#include <memory>

#ifdef TORCH_AVAILABLE
#include <torch/script.h>
#endif // TORCH_AVAILABLE

#ifdef DEBUG_MODE
#include <ros/console.h>
#include "library_crop_detection/stop_watch.hpp"
#endif // DEBUG_MODE

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "library_crop_detection/torch_common.hpp"


namespace igg {

namespace fs = boost::filesystem;

PytorchNetwork::PytorchNetwork(
    const NetworkParameters& kNetworkParameters,
    const SemanticLabelerParameters& kSemanticLabelerParameters,
    const StemExtractorParameters& kStemExtractorParameters):
    kMean_{kNetworkParameters.mean},
    kStd_{kNetworkParameters.std},
    kSemanticLabeler_{SemanticLabeler(kSemanticLabelerParameters)},
    kStemExtractor_{StemExtractor(kStemExtractorParameters)},
    kInputWidth_{kNetworkParameters.input_width},
    kInputHeight_{kNetworkParameters.input_height},
    kInputChannels_{kNetworkParameters.input_channels}{
  ASSERT_TORCH_AVAILABLE;
  #ifdef TORCH_AVAILABLE
  // Allocate buffer memory for network input
  this->input_buffer_ = malloc(4*this->kInputWidth_*this->kInputHeight_*this->kInputChannels_); // 4 bytes per float
  #endif // TORCH_AVAILABLE
}

PytorchNetwork::~PytorchNetwork() {
  // Free buffer memory
  if (this->input_buffer_) {free(this->input_buffer_);}
}

void PytorchNetwork::Infer(NetworkOutput& result, const cv::Mat& kImage) {
  #ifdef TORCH_AVAILABLE

  if(!this->is_loaded_){
    throw std::runtime_error("No module loaded.");
  }

  #ifdef DEBUG_MODE
  // measure inference time in debug mode
  StopWatch stop_watch;
  stop_watch.Start();
  #endif // DEBUG_MODE

  // resize to network input size
  cv::Mat input;
  cv::resize(kImage, input, cv::Size(this->kInputWidth_, this->kInputHeight_));

  // return resized image as well
  std::memcpy(result.ServeInputImageBuffer(this->kInputWidth_, this->kInputHeight_),
      input.ptr(), 4*this->kInputWidth_*this->kInputHeight_);

  // to float
  input.convertTo(input, CV_32F);

  // normalize and put into buffer
  input.forEach<cv::Vec4f>(
    [&](cv::Vec4f& pixel, const int* position) {
      for(int channel_index=0; channel_index<this->kInputChannels_; channel_index++) {
        pixel[channel_index] = (pixel[channel_index]/255.0-this->kMean_[channel_index])/this->kStd_[channel_index];
        const int kBufferIndex = this->kInputHeight_*this->kInputWidth_*channel_index+position[0]*this->kInputWidth_+position[1];
        (static_cast<float*>(this->input_buffer_))[kBufferIndex] = pixel[channel_index];
      }
    }
  );

  torch::Tensor input_tensor = torch::from_blob(this->input_buffer_, {
      1, this->kInputChannels_, this->kInputHeight_, this->kInputWidth_}).to(torch::kFloat32);
  std::vector<torch::jit::IValue> inputs{input_tensor};

  // forward pass
  auto output = this->module_.forward(inputs);

  torch::Tensor semantic_output_tensor = output.toTuple()->elements()[0].toTensor();
  torch::Tensor keypoint_output_tensor = output.toTuple()->elements()[1].toTensor();
  torch::Tensor keypoint_offset_output_tensor = output.toTuple()->elements()[2].toTensor();

  const int kSemanticOutputHeight = semantic_output_tensor.sizes()[2];
  const int kSemanticOutputWidth = semantic_output_tensor.sizes()[3];

  const int kKeypointOutputHeight = keypoint_output_tensor.sizes()[2];
  const int kKeypointOutputWidth = keypoint_output_tensor.sizes()[3];

  const int kOffsetOutputHeight = keypoint_offset_output_tensor.sizes()[2];
  const int kOffsetOutputWidth = keypoint_offset_output_tensor.sizes()[3];

  for(int class_index=0; class_index<semantic_output_tensor.size(1); class_index++) {
    std::memcpy(result.ServeSemanticClassConfidenceBuffer(class_index,
          kSemanticOutputWidth, kSemanticOutputHeight),
                semantic_output_tensor.slice(1, class_index, class_index+1).data_ptr<float>(),
                4*kSemanticOutputWidth*kSemanticOutputHeight);
  }

  std::memcpy(result.ServeStemKeypointConfidenceBuffer(kKeypointOutputWidth, kKeypointOutputHeight),
              keypoint_output_tensor.data_ptr<float>(),
              4*kKeypointOutputWidth*kKeypointOutputHeight);

  std::memcpy(result.ServeStemOffsetXBuffer(kOffsetOutputWidth, kOffsetOutputHeight),
              keypoint_offset_output_tensor.slice(1, 0, 1).data_ptr<float>(),
              4*kOffsetOutputWidth*kOffsetOutputHeight);

  std::memcpy(result.ServeStemOffsetYBuffer(kOffsetOutputWidth, kOffsetOutputHeight),
              keypoint_offset_output_tensor.slice(1, 1, 2).data_ptr<float>(),
              4*kOffsetOutputWidth*kOffsetOutputHeight);

  // postprocessing
  this->kSemanticLabeler_.Infer(result);
  this->kStemExtractor_.Infer(result);

  #ifdef DEBUG_MODE
  double inference_time = stop_watch.ElapsedTime();
  ROS_INFO("Network inference time (including data transfer): %f ms (%f fps)", 1000.0*inference_time, 1.0/inference_time);

  // also write inference times to file
  std::string log_file_name = "/tmp/"+fs::path(this->filepath_).stem().string()
                              +"_pytorch_network_inference_times.txt";
  std::ifstream log_file_in;
  std::ofstream log_file;
  log_file_in.open(log_file_name);
  if (!log_file_in.good()) {
    // does not exists yet, write header
    log_file.open(log_file_name);
    log_file << "#   inference time [ms]   fps [s^-1]   (Torch)\n";
    log_file.close();
  }

  log_file.open(log_file_name, std::ofstream::app);
  log_file << 1000.0*inference_time << "   " << 1.0/inference_time << "\n";
  log_file.close();
  #endif // DEBUG_MODE

  #endif // TORCH_AVAILABLE
}


bool PytorchNetwork::IsReadyToInfer() const {
  return this->is_loaded_;
}

void PytorchNetwork::Load(const std::string& kFilepath, const bool kForceRebuild){
  #ifdef DEBUG_MODE
  // remember filepath for debug purposes
  this->filepath_ = kFilepath;
  #endif // DEBUG_MODE

  #ifdef TORCH_AVAILABLE
  try {
    std::cout << "Load model from: " << kFilepath << "\n";
    this->module_ = torch::jit::load(kFilepath);
    this->module_.eval();
    this->is_loaded_ = true;
  } catch (const c10::Error& kError) {
    this->is_loaded_ = false;
    throw std::runtime_error("Error loading model: "+std::string(kError.what()));
  }
  #endif // TORCH_AVAILABLE
}

int PytorchNetwork::InputWidth() const {return this->kInputWidth_;}


int PytorchNetwork::InputHeight() const {return this->kInputHeight_;}


int PytorchNetwork::InputChannels() const {return this->kInputChannels_;}

}
