/**
 * @file test_tensorrt.cpp
 *
 * Reference: https://github.com/PRBonn/bonnetal/blob/master/deploy/src/segmentation/lib/src/netTensorRT.cpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */
#include <string>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda_runtime_api.h>

// TensorRT stuff
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"

#include "common.h" // CUDA error handling


namespace fs = boost::filesystem;

// logger for TensorRT operations
class Logger: public nvinfer1::ILogger {
public:
  nvinfer1::ILogger::Severity getReportableSeverity() {
    return nvinfer1::ILogger::Severity::kINFO;
  }

private:
  void log(nvinfer1::ILogger::Severity severity, const char* kMessage) override {
    // suppress info-level messages
    if (severity != nvinfer1::ILogger::Severity::kINFO) {
      std::cout << kMessage << "\n";
    }
  }

} logger;


// get the path where our model files are stored
fs::path path_to_models_dir() {
  const char* kCstrModelsDir = std::getenv("M26_G1_SEMANTICS_MODELS_DIR");
  if(!kCstrModelsDir) {
    throw std::runtime_error("Environment varibale M26_G1_SEMANTICS_MODELS_DIR not set.");
  }
  return fs::path(kCstrModelsDir);
}


// get engine from .onnx file
nvinfer1::ICudaEngine* parse_model() {
  const fs::path kPathToOnnxFile = path_to_models_dir()/"simple_unet.onnx";

  auto builder = nvinfer1::createInferBuilder(logger);
  const auto explicitBatch = 1U<<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = builder->createNetworkV2(explicitBatch);

  auto parser = nvonnxparser::createParser(*network, logger);
  auto parsed_network = parser->parseFromFile(kPathToOnnxFile.string().c_str(),
                                              static_cast<int>(logger.getReportableSeverity()));

  builder->setMaxBatchSize(1);
  auto config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1<<20);
  auto engine = builder->buildEngineWithConfig(*network, *config);

  parser->destroy();
  network->destroy();
  config->destroy();
  builder->destroy();

  // store model to disk
  nvinfer1::IHostMemory* serialized_model = engine->serialize();

  //if (!serialized_model) {
  //
  //}

  std::ofstream serialized_model_output_file((path_to_models_dir()/"simple_unet.engine").string(), std::ios::binary);
  serialized_model_output_file.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size());
  serialized_model->destroy();

  return engine;
}


// get engine from serialized model written to disk
nvinfer1::ICudaEngine* read_serialized_model() {
  std::ifstream model_file((path_to_models_dir()/"simple_unet.engine").string(), std::ios::binary);
  if (!model_file) {
    return nullptr;
  }

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

  auto runtime = nvinfer1::createInferRuntime(logger);
  auto engine = runtime->deserializeCudaEngine(model_memory, kModelSize, nullptr);

  free(model_memory);

  return engine;
}


int main () {
  nvinfer1::ICudaEngine* engine = read_serialized_model();

  if (!engine) {
    // need to parse model
    engine = parse_model();
  }

  //if (!engine) {

  //}

  size_t num_bindings = engine->getNbBindings();
  // we expect to have 4 bindings, 1 input and 3 outputs

  //if(num_bindings!=4) {
  //
  //}

  // determine size of network inputs and ouputs and allocate memory on host and device
  std::vector<void*> host_buffer;
  std::vector<void*> device_buffer;

  host_buffer.reserve(num_bindings);
  device_buffer.reserve(num_bindings);

  for (size_t binding_index=0; binding_index<num_bindings; binding_index++) {
    const auto kBindingDimensions = engine->getBindingDimensions(binding_index);
    const auto kBindingDataType = engine->getBindingDataType(binding_index);
    const auto kBindingName = engine->getBindingName(binding_index);

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

  auto context = engine->createExecutionContext();

  // if (!context) {
  //
  // }

  const size_t kInputBindingIndex = engine->getBindingIndex("input");
  const size_t kSemanticOutputBindingIndex = engine->getBindingIndex("semantic_output");

  const auto kInputBindingDimensions = engine->getBindingDimensions(kInputBindingIndex);

  const int kInputWidth = kInputBindingDimensions.d[3];
  const int kInputHeight = kInputBindingDimensions.d[2];
  const int kInputChannels = kInputBindingDimensions.d[1];
  const int kInputSize = kInputWidth*kInputHeight*kInputChannels;

  const auto kSemanticOutputBindingDimensions = engine->getBindingDimensions(kSemanticOutputBindingIndex);

  const int kSemanticOutputWidth = kSemanticOutputBindingDimensions.d[3];
  const int kSemanticOutputHeight = kSemanticOutputBindingDimensions.d[2];
  const int kSemanticOutputChannels = kSemanticOutputBindingDimensions.d[1];
  const int kSemanticOutputSize = kSemanticOutputHeight*kSemanticOutputWidth*kSemanticOutputChannels;

  // read a test image
  cv::Mat input_rgb = cv::imread("../test_data/test_rgb.png", cv::IMREAD_UNCHANGED);
  cv::Mat input_nir = cv::imread("../test_data/test_nir.png", cv::IMREAD_UNCHANGED);
  cv::imshow("input_rgb", input_rgb);
  cv::imshow("input_nir", input_nir);
  cv::cvtColor(input_rgb, input_rgb, cv::COLOR_BGR2RGB);

  // merge tp 4 channels
  std::vector<cv::Mat> channels;
  cv::split(input_rgb, channels);
  channels.emplace_back(input_nir);
  cv::Mat input;
  cv::merge(channels, input);

  // resize
  cv::resize(input, input, cv::Size(kInputWidth, kInputHeight));

  // to float
  input.convertTo(input, CV_32F);

  // normalize and put into buffer

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
}
