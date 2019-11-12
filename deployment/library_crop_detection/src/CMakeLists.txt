add_executable(test_tensorrt test_tensorrt.cpp)
target_link_libraries(test_tensorrt
                      ${OpenCV_LIBS}
                      Boost::filesystem
                      ${CUDA_LIBRARIES}
                      ${NVINFER_LIBRARY}
                      ${NVONNXPARSER_LIBRARY})
