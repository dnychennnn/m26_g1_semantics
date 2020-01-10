/*!
 * @file pytorch_network.cpp
 *
 * @author Yung-Yu Chen
 * @version 0.1
 */

//#include "library_crop_detection/pytorch_network.hpp"

//#include <iostream>
//#include <memory>

//#ifdef TORCH_AVAILABLE
//#include <torch/script.h>
//#endif // TORCH_AVAILABLE

//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>


//namespace igg {

//PytorchNetwork::PytorchNetwork():
  //mean_{0.386, 0.227, 0.054, 0.220}, std_{0.124, 0.072, 0.0108, 0.066} {
  //// TODO get rid of hard-coded normalization

  //#ifdef TORCH_AVAILABLE
  //// Allocate buffer memory for network input
  //this->input_buffer_ = malloc(4*this->input_width_*this->input_height_*this->input_channels_); // 4 bytes per float
  //#endif // TORCH_AVAILABLE

//}

//PytorchNetwork::~PytorchNetwork() {
  //// Free buffer memory
  //free(this->input_buffer_);
//}

//// PytorchNetwork::PytorchNetwork(const NetworkParameters& kParameters):
////     mean_{kParameters.mean}, std_{kParameters.std}, kStemInference_{OpencvStemInference({kParameters})} {
////   this->module_ = new torch::jit::script::Module();
//// }

//void PytorchNetwork::Infer(NetworkInference* result, const cv::Mat& kImage, const bool kMinimalInference) {
  //#ifdef TORCH_AVAILABLE

  //// TODO this whole part should depend on how the model is traced (on cuda or on cpu, refer to self.device in python training part)

  ////if(!this->module_){
    ////throw std::runtime_error("No module loaded.");
  ////}

  //// define cuda device
  //// torch::Device device = torch::kCUDA;

  //// set no grad
  //// torch::NoGradGuard no_grad;

  //// resize to network input size
  //cv::Mat input;
  //cv::resize(kImage, input, cv::Size(this->input_width_, this->input_height_));

  //// return resized image as well
  //std::memcpy(result->ServeInputImageBuffer(this->input_width_, this->input_height_),
      //input.ptr(), 4*this->input_width_*this->input_height_);

  //// TODO maybe have this as class attributes
  //const int kOutputWidth = this->input_width_;
  //const int kOutputHeight = this->input_height_;

  //// to float
  //input.convertTo(input, CV_32F);

  //// normalize and put into buffer
  //// TODO read these from a config
  //input.forEach<cv::Vec4f>(
    //[&](cv::Vec4f& pixel, const int* position) {
      //for(int channel_index=0; channel_index<this->input_channels_; channel_index++) {
        //pixel[channel_index] = (pixel[channel_index]/255.0-this->mean_[channel_index])/this->std_[channel_index];
        //const int kBufferIndex = this->input_height_*this->input_width_*channel_index+position[0]*this->input_width_+position[1];
        //(static_cast<float*>(this->input_buffer_))[kBufferIndex] = pixel[channel_index];
      //}
    //}
  //);

  //// transfer cv mat to tensor( also put into cuda device here)

  //torch::Tensor input_tensor = torch::from_blob(this->input_buffer_, {1, this->input_channels_, this->input_height_, this->input_width_}).to(torch::kFloat32);

  //// put tensor into forward input type
  //// this->module_.to(device);

  //std::vector<torch::jit::IValue> inputs{input_tensor};
  ////inputs.push_back(input_tensor);

  ////std::cout << kImage_tensor << std::endl;

  //// forward
  //auto output = this->module_.forward(inputs);

  //torch::Tensor semantic_output_tensor = output.toTuple()->elements()[0].toTensor();
  //torch::Tensor keypoint_output_tensor = output.toTuple()->elements()[1].toTensor();
  //torch::Tensor keypoint_offset_output_tensor = output.toTuple()->elements()[2].toTensor();

  //// TODO assert output has the expected shape

  //for(int class_index=0; class_index<semantic_output_tensor.size(1); class_index++) {
    //std::memcpy(result->ServeSemanticClassConfidenceBuffer(class_index, this->input_width_, this->input_height_),
                //semantic_output_tensor.slice(1, class_index, class_index+1).data_ptr<float>(),
                //4*this->input_width_*this->input_height_);
  //}

  //std::memcpy(result->ServeStemKeypointConfidenceBuffer(this->input_width_, this->input_height_),
              //keypoint_output_tensor.data_ptr<float>(),
              //4*this->input_width_*this->input_height_);

  //std::memcpy(result->ServeStemOffsetXBuffer(this->input_width_, this->input_height_),
              //keypoint_offset_output_tensor.slice(1, 0, 1).data_ptr<float>(),
              //4*this->input_width_*this->input_height_);

  //std::memcpy(result->ServeStemOffsetYBuffer(this->input_width_, this->input_height_),
              //keypoint_offset_output_tensor.slice(1, 1, 2).data_ptr<float>(),
              //4*this->input_width_*this->input_height_);

  //// postprocessing
  //this->kStemInference_.Infer(result);

  //#endif // TORCH_AVAILABLE
//}


//bool PytorchNetwork::IsReadyToInfer() const {
  //// TODO
  //return true;
//}

//void PytorchNetwork::Load(const std::string& kFilepath, const bool kForceRebuild){
  //#ifdef TORCH_AVAILABLE

  //try {
    //// deserialize the ScriptModule from a file using torch::jit::load().
    //std::cout << "Loading model from " << kFilepath << "." << std::endl;
    //this->module_ = torch::jit::load(kFilepath);
    //this->module_.eval();
  //}

  //catch (const c10::Error& kError) {
    //std::cerr << "Error loading the model." << std::endl;
    //std::cerr << kError.what() << std::endl;
  //}

  //std::cout << "Succeed.\n";

  //#endif // TORCH_AVAILABLE
//}

//int PytorchNetwork::InputWidth() const {return this->input_width_;}


//int PytorchNetwork::InputHeight() const {return this->input_height_;}


//int PytorchNetwork::InputChannels() const {return this->input_channels_;}

//}
