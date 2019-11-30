add_library(crop_detection SHARED tensorrt_network.cpp pytorch_network.cpp network_inference.cpp opencv_stem_inference.cpp)
target_link_libraries(crop_detection ${OpenCV_LIBS} Boost::filesystem)

if(TENSORRT_AVAILABLE)
  target_link_libraries(crop_detection ${CUDA_LIBRARIES} ${NVINFER_LIBRARY})
endif(TENSORRT_AVAILABLE)

if(TENSORRT_ONNX_PARSER_AVAILABLE)
  target_link_libraries(crop_detection ${NVONNXPARSER_LIBRARY})
endif(TENSORRT_ONNX_PARSER_AVAILABLE)

add_executable(network_demo network_demo.cpp)
target_link_libraries(network_demo crop_detection)

