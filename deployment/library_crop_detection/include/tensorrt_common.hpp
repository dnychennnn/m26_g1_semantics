#include <exception>

#ifndef TENSORRT_AVAILABLE
#define ASSERT_TENSORRT_AVAILABLE throw std::runtime_error("TensorRT is not available in this build.")
#else
#define ASSERT_TENSORRT_AVAILABLE
#endif // TENSORRT_AVAILABLE
