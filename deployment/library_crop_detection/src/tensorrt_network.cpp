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

TensorrtNetwork::TensorrtNetwork() {}


NetworkInference TensorrtNetwork::Infer(const cv::Mat& kImage) const {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE

  if (!this->engine_) {
    throw std::runtime_error("No engine loaded.");
  }

  // determine size of network inputs and ouputs and allocate memory on host and device
  // TODO move this in a separate function and only do this once
  const int kNumBindings = this->engine_->getNbBindings();
  // we expect to have 4 bindings, 1 input and 3 outputs

  //if(kNumBindings!=4) {
  //
  //}
  std::vector<void*> host_buffer;

  std::vector<void*> device_buffer;

  host_buffer.reserve(kNumBindings);
  device_buffer.reserve(kNumBindings);

  for (size_t binding_index=0; binding_index<kNumBindings; binding_index++) {
    const auto kBindingDimensions = this->engine_->getBindingDimensions(binding_index);
    const auto kBindingDataType = this->engine_->getBindingDataType(binding_index);
    const auto kBindingName = this->engine_->getBindingName(binding_index);

    if (kBindingDataType!=nvinfer1::DataType::kFLOAT) {
      std::cout << "Not expected data type for binding '" << kBindingName << "'.\n";
    }

    // Determine size by multiplying all dimensions
    size_t binding_size = 1;
    for (size_t dimension_index=0; dimension_index<kBindingDimensions.nbDims; dimension_index++) {
      binding_size *= kBindingDimensions.d[dimension_index];
    }

    std::cout << "Found binding '" << kBindingName << "' of shape (";
    for (size_t dimension_index=0; dimension_index<kBindingDimensions.nbDims; dimension_index++) {
      std::cout << kBindingDimensions.d[dimension_index] << ",";
    }
    std::cout << ") and total size " << binding_size << ".\n";

    // allocate host and device memory according to the determined binding size
    HANDLE_ERROR(cudaMalloc(&device_buffer[binding_index], 4*binding_size)); // 4 bytes per float
    HANDLE_ERROR(cudaMallocHost(&host_buffer[binding_index], 4*binding_size)); // 4 bytes per float
  }

  auto context = this->engine_->createExecutionContext();

  // if (!context) {
  //
  // }

  const size_t kInputBindingIndex = this->engine_->getBindingIndex("input");
  const size_t kSemanticOutputBindingIndex = this->engine_->getBindingIndex("semantic_output");

  const auto kInputBindingDimensions = this->engine_->getBindingDimensions(kInputBindingIndex);

  const int kInputWidth = kInputBindingDimensions.d[3];
  const int kInputHeight = kInputBindingDimensions.d[2];
  const int kInputChannels = kInputBindingDimensions.d[1];
  const int kInputSize = kInputWidth*kInputHeight*kInputChannels;

  const auto kSemanticOutputBindingDimensions = this->engine_->getBindingDimensions(kSemanticOutputBindingIndex);

  const int kSemanticOutputWidth = kSemanticOutputBindingDimensions.d[3];
  const int kSemanticOutputHeight = kSemanticOutputBindingDimensions.d[2];
  const int kSemanticOutputChannels = kSemanticOutputBindingDimensions.d[1];
  const int kSemanticOutputSize = kSemanticOutputHeight*kSemanticOutputWidth*kSemanticOutputChannels;

  // resize to network input size
  cv::Mat input;
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
        const int kBufferIndex = kInputHeight*kInputWidth*channel_index+position[0]*kInputWidth+position[1];
        (static_cast<float*>(host_buffer[kInputBindingIndex]))[kBufferIndex] = pixel[channel_index];
    }
    }
  );

  // TODO use asynchronus memcopy as in bonnetal

  // transfer to device
  HANDLE_ERROR(cudaMemcpy(device_buffer[kInputBindingIndex],
                          host_buffer[kInputBindingIndex],
                          4*kInputSize,
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaDeviceSynchronize());

  // pass through network
  context->executeV2(&device_buffer[0]); // version 2 is without batch size
  HANDLE_ERROR(cudaDeviceSynchronize());

  // transfer back to host
  HANDLE_ERROR(cudaMemcpy(host_buffer[kSemanticOutputBindingIndex],
                          device_buffer[kSemanticOutputBindingIndex],
                          4*kSemanticOutputSize,
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());

  // get confidences as cv::Mat
  cv::Mat semantic_output = cv::Mat::zeros(cv::Size(kSemanticOutputWidth, kSemanticOutputHeight), CV_32FC3);

  semantic_output.forEach<cv::Vec3f>(
    [&](cv::Vec3f& pixel, const int* position) {
      for(int channel_index=0; channel_index<kSemanticOutputChannels; channel_index++) {
        const int kBufferIndex = kSemanticOutputHeight*kSemanticOutputWidth*channel_index+position[0]*kSemanticOutputWidth+position[1];
        pixel[channel_index] = static_cast<float*>(host_buffer[kSemanticOutputBindingIndex])[kBufferIndex];
      }
    }
  );

  std::vector<cv::Mat> semantic_class_confidences;
  cv::split(semantic_output, semantic_class_confidences);

  cv::imshow("sugar beet confidence", semantic_class_confidences[0]);
  cv::waitKey(0);

  return NetworkInference();
  // TODO postprocessing, put everything in a NetworkInference instance

  #endif // TENSORRT_AVAILABLE
}


bool TensorrtNetwork::IsReadyToInfer() const {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE
  if (this->engine_) {return true;}
  return false;
  #endif // TENSORRT_AVAILABLE
}


bool TensorrtNetwork::Load(const std::string kFilepath, const bool kForceRebuild) {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE

  std::string kEngineFilepath = (fs::path(kFilepath).parent_path()/(fs::path(kFilepath).stem().string()+".engine")).string();

  if (!kForceRebuild) {
    // try to use previous engine, same filename with '.engine' extension
    if (this->LoadSerialized(kEngineFilepath)) {return true;}
  }

  std::ifstream model_file(kFilepath, std::ios::binary);
  if (!model_file) {
    std::cerr << "Cannnot read model file " << kFilepath << "\n";
    return false;
  }

  auto builder = nvinfer1::createInferBuilder(this->logger_);
  const auto explicitBatch = 1U<<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = builder->createNetworkV2(explicitBatch);

  auto parser = nvonnxparser::createParser(*network, this->logger_);
  auto parsed_network = parser->parseFromFile(kFilepath.c_str(),
                                              static_cast<int>(this->logger_.getReportableSeverity()));

  builder->setMaxBatchSize(1);
  auto config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1<<20);

  if(this->engine_){this->engine_->destroy();}
  this->engine_ = builder->buildEngineWithConfig(*network, *config);

  parser->destroy();
  network->destroy();
  config->destroy();
  builder->destroy();

  // store model to disk
  auto serialized_model = this->engine_->serialize();

  //if (!serialized_model) {
  //
  //}

  std::ofstream serialized_model_output_file(kEngineFilepath, std::ios::binary);
  serialized_model_output_file.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size());
  serialized_model->destroy();

  return true;
  #endif // TENSORRT_AVAILABLE
}


bool TensorrtNetwork::LoadSerialized(const std::string kFilepath) {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE

  std::ifstream model_file(kFilepath, std::ios::binary);
  if (!model_file) {
    std::cerr << "Cannnot read model engine " << kFilepath << "\n";
    return false;
  }

  std::cout << "Load serialized engine from " << kFilepath << "\n";

  std::stringstream model_stream;
  model_stream.seekg(0, model_stream.beg);
  model_stream << model_file.rdbuf();
  model_file.close();

  model_stream.seekg(0, std::ios::end);
  const size_t kModelSize = model_stream.tellg();
  model_stream.seekg(0, std::ios::beg);

  void* model_memory = malloc(kModelSize);

  // if (!model_memory) {
  //
  // }

  model_stream.read(reinterpret_cast<char*>(model_memory), kModelSize);

  auto runtime = nvinfer1::createInferRuntime(this->logger_);
  if(this->engine_){this->engine_->destroy();}
  this->engine_ = runtime->deserializeCudaEngine(model_memory, kModelSize, nullptr);

  free(model_memory);
  #endif // TENSORRT_AVAILABLE
}


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
