/**
 * @file test_tensorrt.cpp
 *
 * @author Jan Quakernack
 * @version 0.1
 */
#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <boost/filesystem.hpp>

#include <cuda_runtime_api.h>

// TensorRT stuff
#include "NvInfer.h"
#include "NvOnnxParser.h"


namespace fs = boost::filesystem;


fs::path path_to_models_dir() {
  const char* kCstrModelsDir = std::getenv("M26_G1_SEMANTICS_MODELS_DIR");
  if(!kCstrModelsDir) {
    throw std::runtime_error("Environment varibale M26_G1_SEMANTICS_MODELS_DIR not set.");
  }
  return fs::path(kCstrModelsDir);
}


class Logger: public nvinfer1::ILogger {
public:
  nvinfer1::ILogger::Severity getReportableSeverity() {
    return nvinfer1::ILogger::Severity::kINFO;
  }

private:
  void log(nvinfer1::ILogger::Severity severity, const char* kMessage) override {
    // suppress info-level messages
    //if (severity != nvinfer1::ILogger::Severity::kINFO) {
      //std::cout << msg << "\n";
    //}
    std::cout << kMessage << "\n";
  }


} gLogger;


int main () {
  const fs::path kPathToOnnxFile = path_to_models_dir()/"simple_unet.onnx";
  std::cout << kPathToOnnxFile << "\n";

  auto builder = nvinfer1::createInferBuilder(gLogger);
  auto network = builder->createNetwork();
  auto parser = nvonnxparser::createParser(*network, gLogger);

  auto parsed_network = parser->parseFromFile(kPathToOnnxFile.string().c_str(),
                                         static_cast<int>(gLogger.getReportableSeverity()));

}
