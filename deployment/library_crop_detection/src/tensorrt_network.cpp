/*!
 * @file tensorrt_network.cpp
 *
 * Reference: https://github.com/PRBonn/bonnetal/blob/master/deploy/src/segmentation/lib/src/netTensorRT.cpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */

#include "tensorrt_network.hpp"

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#ifdef TENSORRT_AVAILABLE
#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#endif // TENSORRT_AVAILABLE

#include "handle_cuda_error.hpp"


namespace igg {

namespace fs = boost::filesystem;

TensorrtNetwork::TensorrtNetwork(
    const std::vector<float>& kMean, const std::vector<float>& kStd):
    mean_{kMean}, std_{kStd} {}


TensorrtNetwork::~TensorrtNetwork() {
  if(this->engine_){this->engine_->destroy();}
  this->FreeBufferMemory();
}


void TensorrtNetwork::Infer(NetworkInference* result, const cv::Mat& kImage, const bool kMinimalInference) {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE

  if (!this->engine_) {
    throw std::runtime_error("No engine loaded.");
  }

  auto context = this->engine_->createExecutionContext();
  if (!context) {
    throw std::runtime_error("Could not create execution context.");
  }

  // resize to network input size
  cv::Mat input;
  cv::resize(kImage, input, cv::Size(this->input_width_, this->input_height_));

  // to float
  input.convertTo(input, CV_32F);

  // normalize and put into buffer
  input.forEach<cv::Vec4f>(
    [&](cv::Vec4f& pixel, const int* position) {
      for(int channel_index=0; channel_index<this->input_channels_; channel_index++) {
        pixel[channel_index] = (pixel[channel_index]/255.0-this->mean_[channel_index])/this->std_[channel_index];
        const int kBufferIndex = this->input_height_*this->input_width_*channel_index+position[0]*this->input_width_+position[1];
        (static_cast<float*>(this->host_buffers_[this->input_binding_index_]))[kBufferIndex] = pixel[channel_index];
      }
    }
  );

  // TODO use asynchronus memcopy as in bonnetal

  // transfer to device
  HANDLE_ERROR(cudaMemcpy(this->device_buffers_[this->input_binding_index_],
                          this->host_buffers_[this->input_binding_index_],
                          4*this->input_width_*this->input_height_*this->input_channels_,
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaDeviceSynchronize());



  // pass through network
  context->executeV2(&((this->device_buffers_)[this->input_binding_index_])); // version 2 is without batch size
  HANDLE_ERROR(cudaDeviceSynchronize());

  // transfer back to host
  HANDLE_ERROR(cudaMemcpy(this->host_buffers_[this->semantic_output_binding_index_],
                          this->device_buffers_[this->semantic_output_binding_index_],
                          4*this->semantic_output_width_*this->semantic_output_height_*this->semantic_output_channels_,
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());

  // get confidences as cv::Mat
  cv::Mat semantic_output = cv::Mat::zeros(cv::Size(this->semantic_output_width_, this->semantic_output_height_), CV_32FC3);

  semantic_output.forEach<cv::Vec3f>(
    [&](cv::Vec3f& pixel, const int* position) {
      for(int channel_index=0; channel_index<this->semantic_output_channels_; channel_index++) {
        const int kBufferIndex = this->semantic_output_height_*this->semantic_output_width_*channel_index+position[0]*this->semantic_output_width_+position[1];
        pixel[channel_index] = static_cast<float*>(this->host_buffers_[this->semantic_output_binding_index_])[kBufferIndex];
      }
    }
  );

  std::vector<cv::Mat> semantic_class_confidences;
  cv::split(semantic_output, semantic_class_confidences);

  cv::imshow("background confidence", semantic_class_confidences[0]);
  cv::imshow("weed confidence", semantic_class_confidences[1]);
  cv::imshow("sugar beet confidence", semantic_class_confidences[2]);
  cv::waitKey(0);

  // TODO postprocessing, put everything in a NetworkInference instance

  context->destroy();

  #endif // TENSORRT_AVAILABLE
}


bool TensorrtNetwork::IsReadyToInfer() const {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE
  if (this->engine_) {return true;}
  return false;
  #endif // TENSORRT_AVAILABLE
}


void TensorrtNetwork::Load(const std::string& kFilepath, const bool kForceRebuild) {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE

  std::string kEngineFilepath = (fs::path(kFilepath).parent_path()/(fs::path(kFilepath).stem().string()+".engine")).string();

  if (kForceRebuild || !this->LoadSerialized(kEngineFilepath)) {
    // not okay to use previous engine, parse (again)

    std::ifstream model_file(kFilepath, std::ios::binary);
    if (!model_file) {
      throw std::runtime_error("Cannnot read model file: \""+kFilepath+"\"");
    }

    auto builder = nvinfer1::createInferBuilder(this->logger_);
    const auto explicitBatch = 1U<<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);

    auto parser = nvonnxparser::createParser(*network, this->logger_);
    if(!parser->parseFromFile(kFilepath.c_str(),
         static_cast<int>(this->logger_.getReportableSeverity()))){
      throw std::runtime_error("Cannnot parse model file: \""+kFilepath+"\"");
    }

    builder->setMaxBatchSize(1);
    auto config = builder->createBuilderConfig();
    // config->setMaxWorkspaceSize(1<<20); // maximum amount of GPU memory we allow the engine to use

    if(this->engine_){this->engine_->destroy();}
    this->engine_ = builder->buildEngineWithConfig(*network, *config);

    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();

    // store model to disk
    auto serialized_model = this->engine_->serialize();
    if (!serialized_model) {
      std::cerr << "Cannot serialize engine.\n";
    } else {
      std::ofstream serialized_model_output_file(kEngineFilepath, std::ios::binary);
      serialized_model_output_file.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size());
      serialized_model->destroy();
    }
  }
  this->ReadBindingsAndAllocateBufferMemory();
  #endif // TENSORRT_AVAILABLE
}


bool TensorrtNetwork::LoadSerialized(const std::string& kFilepath) {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE

  std::ifstream model_file(kFilepath, std::ios::binary);
  if (!model_file) {
    std::cout << "Cannnot read model engine: \"" << kFilepath << "\"\n";
    return false;
  }

  std::cout << "Load serialized engine: \"" << kFilepath << "\"\n";

  std::stringstream model_stream;
  model_stream.seekg(0, model_stream.beg);
  model_stream << model_file.rdbuf();
  model_file.close();

  model_stream.seekg(0, std::ios::end);
  const size_t kModelSize = model_stream.tellg();
  model_stream.seekg(0, std::ios::beg);

  void* model_memory = malloc(kModelSize);
  if (!model_memory) {
    std::cerr << "Cannot allocate memory to load serialized model.\n";
    return false;
  }

  model_stream.read(reinterpret_cast<char*>(model_memory), kModelSize);

  auto runtime = nvinfer1::createInferRuntime(this->logger_);
  if(this->engine_){this->engine_->destroy();}
  this->engine_ = runtime->deserializeCudaEngine(model_memory, kModelSize, nullptr);

  free(model_memory);
  return true;
  #endif // TENSORRT_AVAILABLE
}


void TensorrtNetwork::ReadBindingsAndAllocateBufferMemory() {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE

  // free whatever is buffered now
  this->FreeBufferMemory();
  // will also clear and shrink the vectors

  // make sure we have an engine loaded
  if (!this->engine_) {
    throw std::runtime_error("No engine loaded.");
  }

  const int kNumBindings = this->engine_->getNbBindings();
  // we expect to have 4 bindings, 1 input and 3 outputs
  if(kNumBindings!=4) {
    throw std::runtime_error("Unexpected number of bindings: "+std::to_string(kNumBindings));
  }

  this->host_buffers_.reserve(kNumBindings);
  this->device_buffers_.reserve(kNumBindings);

  // check the size of each binding
  for (int binding_index=0; binding_index<kNumBindings; binding_index++) {
    const auto kBindingDimensions = this->engine_->getBindingDimensions(binding_index);
    const auto kBindingDataType = this->engine_->getBindingDataType(binding_index);
    const auto kBindingName = this->engine_->getBindingName(binding_index);

    if (kBindingDataType!=nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error("Unexpected data type for binding: "+std::string(kBindingName));
    }

    // Determine size by multiplying all dimensions
    int binding_size = 1;
    for (int dimension_index=0; dimension_index<kBindingDimensions.nbDims; dimension_index++) {
      binding_size *= kBindingDimensions.d[dimension_index];
    }

    // some output for checking
    std::cout << "Found binding '" << kBindingName << "' of shape (";
    for (int dimension_index=0; dimension_index<kBindingDimensions.nbDims; dimension_index++) {
      std::cout << kBindingDimensions.d[dimension_index] << ",";
    }
    std::cout << ") and total size " << binding_size << ".\n";

    // allocate host and device memory according to the determined binding size
    HANDLE_ERROR(cudaMallocHost(&(this->host_buffers_[binding_index]), 4*binding_size)); // 4 bytes per float
    HANDLE_ERROR(cudaMalloc(&(this->device_buffers_[binding_index]), 4*binding_size)); // 4 bytes per float
  }

  this->input_binding_index_ = this->engine_->getBindingIndex("input");
  const auto kInputBindingDimensions = this->engine_->getBindingDimensions(this->input_binding_index_);
  this->input_width_ = kInputBindingDimensions.d[3];
  this->input_height_ = kInputBindingDimensions.d[2];
  this->input_channels_ = kInputBindingDimensions.d[1];

  this->semantic_output_binding_index_ = this->engine_->getBindingIndex("semantic_output");
  const auto kSemanticOutputBindingDimensions = this->engine_->getBindingDimensions(this->semantic_output_binding_index_);
  this->semantic_output_width_ = kSemanticOutputBindingDimensions.d[3];
  this->semantic_output_height_ = kSemanticOutputBindingDimensions.d[2];
  this->semantic_output_channels_ = kSemanticOutputBindingDimensions.d[1];

  this->stem_keypoint_output_binding_index_ = this->engine_->getBindingIndex("stem_keypoint_output");
  const auto kStemKeypointOutputBindingDimensions = this->engine_->getBindingDimensions(this->stem_keypoint_output_binding_index_);
  this->stem_keypoint_output_width_ = kStemKeypointOutputBindingDimensions.d[3];
  this->stem_keypoint_output_height_ = kStemKeypointOutputBindingDimensions.d[2];
  this->stem_keypoint_output_channels_ = kStemKeypointOutputBindingDimensions.d[1];

  this->stem_offset_output_binding_index_ = this->engine_->getBindingIndex("stem_offset_output");
  const auto kStemOffsetOutputBindingDimensions = this->engine_->getBindingDimensions(this->stem_offset_output_binding_index_);
  this->stem_offset_output_width_ = kStemOffsetOutputBindingDimensions.d[3];
  this->stem_offset_output_height_ = kStemOffsetOutputBindingDimensions.d[2];
  this->stem_offset_output_channels_ = kStemOffsetOutputBindingDimensions.d[1];

  if (!(this->input_width_==this->semantic_output_width_
        && this->input_width_==this->stem_keypoint_output_width_
        && this->input_width_==this->stem_offset_output_width_
        && this->input_height_==this->semantic_output_height_
        && this->input_height_==this->stem_keypoint_output_height_
        && this->input_height_==this->stem_offset_output_height_)) {
    throw std::runtime_error("Expect all inputs and ouputs to have the same height and width.");
  }

  if (!(this->input_channels_==this->mean_.size()
        &&this->input_channels_==this->std_.size())){
    throw std::runtime_error("Network input channels do not match the provided normalization parameters.");
  }
  #endif // TENSORRT_AVAILABLE
}


void TensorrtNetwork::FreeBufferMemory() {
  // host
  for(void* buffer: this->host_buffers_) {
    #ifdef TENSORRT_AVAILABLE
    HANDLE_ERROR(cudaFreeHost(buffer));
    #endif // TENSORRT_AVAILABLE
  }
  this->host_buffers_.clear();
  this->host_buffers_.shrink_to_fit();

  // device
  for(void* buffer: this->device_buffers_) {
    #ifdef TENSORRT_AVAILABLE
    HANDLE_ERROR(cudaFree(buffer));
    #endif // TENSORRT_AVAILABLE
  }
  this->device_buffers_.clear();
  this->device_buffers_.shrink_to_fit();
}


int TensorrtNetwork::InputWidth() const {return this->input_width_;}


int TensorrtNetwork::InputHeight() const {return this->input_height_;}


int TensorrtNetwork::InputChannels() const {return this->input_channels_;}


#ifdef TENSORRT_AVAILABLE
nvinfer1::ILogger::Severity TensorrtNetworkLogger::getReportableSeverity() {
  return nvinfer1::ILogger::Severity::kINFO;
}

void TensorrtNetworkLogger::log(nvinfer1::ILogger::Severity severity, const char* kMessage) {
  if (severity != nvinfer1::ILogger::Severity::kINFO) {
    std::cout << kMessage << "\n";
  }
}
#endif // TENSORRT_AVAILABLE

} // namespace igg
