/*!
 * @file tensorrt_network.cpp
 *
 * Reference: https://github.com/PRBonn/bonnetal/blob/master/deploy/src/segmentation/lib/src/netTensorRT.cpp
 *
 * Some parts of this implementation are strongly inspired by the original
 * implementation of bonnetal, in particular the way buffer memory is allocated.
 *
 * @version 0.1
 */

#include "library_crop_detection/tensorrt_network.hpp"

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

#ifdef DEBUG_MODE
#include <ros/console.h>
#include <fstream>
#include "library_crop_detection/stop_watch.hpp"
#endif // DEBUG_MODE

#ifdef CUDA_AVAILABLE
#include "handle_cuda_error.hpp"
#endif // CUDA_AVAILABLE


namespace igg {

namespace fs = boost::filesystem;

TensorrtNetwork::TensorrtNetwork(
    const NetworkParameters& kNetworkParameters,
    const SemanticLabelerParameters& kSemanticLabelerParameters,
    const StemExtractorParameters& kStemExtractorParameters):
    mean_{kNetworkParameters.mean},
    std_{kNetworkParameters.std},
    kSemanticLabeler_{SemanticLabeler(kSemanticLabelerParameters)},
    kStemExtractor_{StemExtractor(kStemExtractorParameters)}
    #ifdef CUDA_AVAILABLE
    , stem_extractor_gpu_{StemExtractorGpu(kStemExtractorParameters)}
    #endif // CUDA_AVAILABLE
{
  ASSERT_TENSORRT_AVAILABLE;
}


TensorrtNetwork::~TensorrtNetwork() {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE
  this->FreeBufferMemory();
  //if(this->engine_exists_){this->engine_->destroy(); this->engine_exists_ = false;}
  #endif // TENSORRT_AVAILABLE
}


void TensorrtNetwork::Infer(NetworkOutput& result, const cv::Mat& kImage) {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE

  if (!this->engine_exists_) {
    throw std::runtime_error("No engine loaded.");
  }

  #ifdef DEBUG_MODE
  // measure inference time in debug mode
  StopWatch stop_watch;
  stop_watch.Start();
  #endif // DEBUG_MODE

  auto context = this->engine_->createExecutionContext(); // allows to run one engine in several contexts
  if (!context) {
    throw std::runtime_error("Could not create execution context.");
  }

  // resize input image to network input size
  cv::Mat input;
  cv::resize(kImage, input, cv::Size(this->input_width_, this->input_height_));

  // return resized image as well
  std::memcpy(result.ServeInputImageBuffer(this->input_width_, this->input_height_),
      input.ptr(), 4*this->input_width_*this->input_height_);

  // to float
  input.convertTo(input, CV_32F);

  // normalize and put into buffer
  input.forEach<cv::Vec4f>(
    [&](cv::Vec4f& pixel, const int* position) {
      for(int channel_index=0; channel_index<this->input_channels_; channel_index++) {
        pixel[channel_index] = (pixel[channel_index]/255.0-this->mean_[channel_index])/this->std_[channel_index];
        const int kBufferIndex = this->input_height_*this->input_width_*channel_index+position[0]*this->input_width_+position[1];
        (static_cast<float*>(this->host_buffer_))[kBufferIndex] = pixel[channel_index];
      }
    }
  );

  // use a separate cuda stream
  cudaStream_t stream;
  HANDLE_ERROR(cudaStreamCreate(&stream));

  // transfer to device
  HANDLE_ERROR(cudaMemcpyAsync(this->device_buffers_[this->input_binding_index_],
                               this->host_buffer_,
                               4*this->input_width_*this->input_height_*this->input_channels_,
                               cudaMemcpyHostToDevice,
                               stream));

  // pass through network
  cudaEvent_t event_input_consumed; // triggered when buffer can be refilled, not used here
  HANDLE_ERROR(cudaEventCreate(&event_input_consumed));
  context->enqueueV2(&((this->device_buffers_)[this->input_binding_index_]), stream, &event_input_consumed); // version 2 is without batch size

  #ifdef CUDA_AVAILABLE

  HANDLE_ERROR(cudaStreamSynchronize(stream));

  this->stem_extractor_gpu_.Infer(
      static_cast<float*>(this->device_buffers_[this->stem_keypoint_output_binding_index_]),
      static_cast<float*>(this->device_buffers_[this->stem_offset_output_binding_index_]),
      result);

  HANDLE_ERROR(cudaStreamSynchronize(stream));

  #endif // CUDA_AVAILABLE

  // retrieve semantic class confidences
  for(int class_index=0; class_index<this->semantic_output_channels_; class_index++) {
    HANDLE_ERROR(cudaMemcpyAsync(result.ServeSemanticClassConfidenceBuffer(class_index, this->semantic_output_width_, this->semantic_output_height_),
                                 this->device_buffers_[this->semantic_output_binding_index_]
                                 +4*class_index*this->semantic_output_height_*this->semantic_output_width_,
                                 4*this->semantic_output_width_*this->semantic_output_height_,
                                 cudaMemcpyDeviceToHost, stream));
  }

  // retrieve stem keypoint confidence
  HANDLE_ERROR(cudaMemcpyAsync(result.ServeStemKeypointConfidenceBuffer(this->stem_keypoint_output_width_, this->stem_keypoint_output_height_),
                               this->device_buffers_[this->stem_keypoint_output_binding_index_],
                               4*this->stem_keypoint_output_width_*this->stem_keypoint_output_height_*this->stem_keypoint_output_channels_,
                               cudaMemcpyDeviceToHost, stream));

  // retrieve stem offset x
  HANDLE_ERROR(cudaMemcpyAsync(result.ServeStemOffsetXBuffer(this->stem_offset_output_width_, this->stem_offset_output_height_),
                               this->device_buffers_[this->stem_offset_output_binding_index_],
                               4*this->stem_offset_output_width_*this->stem_offset_output_height_,
                               cudaMemcpyDeviceToHost, stream));

  // retrieve stem offset y
  HANDLE_ERROR(cudaMemcpyAsync(result.ServeStemOffsetYBuffer(this->stem_offset_output_width_, this->stem_offset_output_height_),
                               this->device_buffers_[this->stem_offset_output_binding_index_]
                               +4*this->stem_offset_output_width_*this->stem_offset_output_height_,
                               4*this->stem_offset_output_width_*this->stem_offset_output_height_,
                               cudaMemcpyDeviceToHost, stream));

  HANDLE_ERROR(cudaStreamSynchronize(stream));
  HANDLE_ERROR(cudaStreamDestroy(stream));

  context->destroy();

  // postprocessing
  this->kSemanticLabeler_.Infer(result);
  //this->kStemExtractor_.Infer(result);

  #ifdef DEBUG_MODE
  double inference_time = stop_watch.ElapsedTime();
  ROS_INFO("Network inference time (including data transfer): %f ms (%f fps)", 1000.0*inference_time, 1.0/inference_time);

  // also write inference times to file
  std::string log_file_name = "/tmp/"+fs::path(this->filepath_).stem().string()
                              +"_tensorrt_network_inference_times.txt";
  std::ifstream log_file_in;
  std::ofstream log_file;
  log_file_in.open(log_file_name);
  if (!log_file_in.good()) {
    // does not exists yet, write header
    log_file.open(log_file_name);
    log_file << "#   inference time [ms]   fps [s^-1]   (TensorRT)\n";
    log_file.close();
  }

  log_file.open(log_file_name, std::ofstream::app);
  log_file << 1000.0*inference_time << "   " << 1.0/inference_time << "\n";
  log_file.close();
  #endif // DEBUG_MODE

  #endif // TENSORRT_AVAILABLE
}


bool TensorrtNetwork::IsReadyToInfer() const {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE
  if (this->engine_exists_) {return true;}
  return false;
  #endif // TENSORRT_AVAILABLE
}


void TensorrtNetwork::Load(const std::string& kFilepath, const bool kForceRebuild) {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE

  #ifdef DEBUG_MODE
  // remember filepath for debug purposes
  this->filepath_ = kFilepath;
  #endif // DEBUG_MODE

  std::string kEngineFilepath = (fs::path(kFilepath).parent_path()/(fs::path(kFilepath).stem().string()+".engine")).string();

  if (kForceRebuild || !this->LoadSerialized(kEngineFilepath)) {
    // not okay to use previous engine, parse (again)

    std::ifstream model_file(kFilepath, std::ios::binary);
    if (!model_file) {
      throw std::runtime_error("Cannnot read model file: '"+kFilepath+"'");
    }

    auto builder = nvinfer1::createInferBuilder(this->logger_);
    const auto explicitBatch = 1U<<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);

    auto parser = nvonnxparser::createParser(*network, this->logger_);
    if(!parser->parseFromFile(kFilepath.c_str(),
         static_cast<int>(this->logger_.getReportableSeverity()))){
      parser->destroy();
      builder->destroy();
      network->destroy();
      throw std::runtime_error("Cannnot parse model file: '"+kFilepath+"'");
    }

    builder->setMaxBatchSize(1);
    auto config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1<<20);

    if (this->engine_exists_) {this->engine_->destroy(); this->engine_exists_ = false;}
    this->engine_ = builder->buildEngineWithConfig(*network, *config);
    this->engine_exists_ = true;

    parser->destroy();
    builder->destroy();
    config->destroy();
    network->destroy();

    // store model to disk
    auto serialized_model = this->engine_->serialize();
    if (!serialized_model) {
      std::cout << "Warning: Cannot serialize engine.\n";
    } else {
      std::ofstream serialized_model_output_file(kEngineFilepath, std::ios::binary);
      serialized_model_output_file.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size());
      serialized_model->destroy();
    }
  }

  this->ReadBindingsAndAllocateBufferMemory();

  #ifdef CUDA_AVAILABLE

  // scaling to give stem positions relative to input size
  const float kScaling = static_cast<float>(this->input_height_)/static_cast<float>(this->stem_offset_output_height_);

  this->stem_extractor_gpu_.LoadAndAllocateBuffers(
      this->stem_keypoint_output_height_,
      this->stem_keypoint_output_width_,
      kScaling);

  #endif // CUDA_AVAILABLE

  #endif // TENSORRT_AVAILABLE
}


bool TensorrtNetwork::LoadSerialized(const std::string& kFilepath) {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE

  std::ifstream model_file(kFilepath, std::ios::binary);
  if (!model_file) {
    std::cout << "Warning: Cannnot read model engine: '" << kFilepath << "'\n";
    model_file.close();
    return false;
  }

  std::cout << "Load serialized engine: '" << kFilepath << "'\n";

  std::stringstream model_stream;
  model_stream.seekg(0, model_stream.beg);
  model_stream << model_file.rdbuf();
  model_file.close();

  model_stream.seekg(0, std::ios::end);
  const size_t kModelSize = model_stream.tellg();
  model_stream.seekg(0, std::ios::beg);

  void* model_memory = malloc(kModelSize);
  if (!model_memory) {
    std::cout << "Warning: Cannot allocate memory to load serialized model.\n";
    return false;
  }

  model_stream.read(reinterpret_cast<char*>(model_memory), kModelSize);

  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(this->logger_);

  if (!runtime) {
    std::cout << "Warning: Cannot not create runtime." << std::endl;
    if (model_memory) {free(model_memory); model_memory=nullptr;}
    return false;
  }

  if (this->engine_exists_) {this->engine_->destroy(); this->engine_exists_ = false;}
  this->engine_ = runtime->deserializeCudaEngine(model_memory, kModelSize, nullptr);
  this->engine_exists_ = true;

  if (model_memory) {free(model_memory); model_memory = nullptr;}
  runtime->destroy();

  return true;
  #endif // TENSORRT_AVAILABLE
}


void TensorrtNetwork::ReadBindingsAndAllocateBufferMemory() {
  ASSERT_TENSORRT_AVAILABLE;
  #ifdef TENSORRT_AVAILABLE
  // free whatever is buffered now
  this->FreeBufferMemory();

  // make sure we have an engine loaded
  if (!this->engine_exists_) {
    throw std::runtime_error("No engine loaded.");
  }

  const int kNumBindings = this->engine_->getNbBindings();
  // we expect to have 4 bindings, 1 input and 3 outputs
  if(kNumBindings!=4) {
    throw std::runtime_error("Unexpected number of bindings: "+std::to_string(kNumBindings));
  }

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
    void* device_buffer;
    HANDLE_ERROR(cudaMalloc(&device_buffer, 4*binding_size)); // 4 bytes per float
    this->device_buffers_.emplace_back(device_buffer);
  }

  this->input_binding_index_ = this->engine_->getBindingIndex("input");
  const auto kInputBindingDimensions = this->engine_->getBindingDimensions(this->input_binding_index_);
  this->input_width_ = kInputBindingDimensions.d[3];
  this->input_height_ = kInputBindingDimensions.d[2];
  this->input_channels_ = kInputBindingDimensions.d[1];

  if (this->input_channels_!=4) {
    throw std::runtime_error("Expected input binding to have four channels.");
  }

  HANDLE_ERROR(cudaMallocHost(&(this->host_buffer_), 4*this->input_width_*this->input_height_*this->input_channels_)); // 4 bytes per float

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

  if (this->stem_keypoint_output_channels_!=1) {
    throw std::runtime_error("Expected binding to have one channel: 'stem_keypoint_output'");
  }

  this->stem_offset_output_binding_index_ = this->engine_->getBindingIndex("stem_offset_output");
  const auto kStemOffsetOutputBindingDimensions = this->engine_->getBindingDimensions(this->stem_offset_output_binding_index_);
  this->stem_offset_output_width_ = kStemOffsetOutputBindingDimensions.d[3];
  this->stem_offset_output_height_ = kStemOffsetOutputBindingDimensions.d[2];
  this->stem_offset_output_channels_ = kStemOffsetOutputBindingDimensions.d[1];

  if (this->stem_offset_output_channels_!=2) {
    throw std::runtime_error("Expected binding to have two channels: 'stem_offset_output'");
  }

  // check if input size equals output size -- this is not the case any longer as we use an initial, strided convolution for downsampling
  // if (!(this->input_width_==this->semantic_output_width_
        //&& this->input_width_==this->stem_keypoint_output_width_
        //&& this->input_width_==this->stem_offset_output_width_
        //&& this->input_height_==this->semantic_output_height_
        //&& this->input_height_==this->stem_keypoint_output_height_
        //&& this->input_height_==this->stem_offset_output_height_)) {
    //throw std::runtime_error("Expect all inputs and ouputs to have the same height and width.");
  //}

  if (!(this->input_channels_==this->mean_.size()
        &&this->input_channels_==this->std_.size())){
    throw std::runtime_error("Network input channels do not match the provided normalization parameters.");
  }
  #endif // TENSORRT_AVAILABLE
}


void TensorrtNetwork::FreeBufferMemory() {
  // host
  #ifdef TENSORRT_AVAILABLE
  if (this->host_buffer_) {
    HANDLE_ERROR(cudaFreeHost(this->host_buffer_)); this->host_buffer_ = nullptr;
  }
  #endif // TENSORRT_AVAILABLE

  // device
  for(size_t buffer_index = 0; buffer_index<this->device_buffers_.size(); buffer_index++) {
    #ifdef TENSORRT_AVAILABLE
    if (this->device_buffers_[buffer_index]) {
      HANDLE_ERROR(cudaFree(this->device_buffers_[buffer_index]));
    }
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
    std::cout << "[TENSORRT] " << kMessage << "\n";
  }
}
#endif // TENSORRT_AVAILABLE

} // namespace igg
