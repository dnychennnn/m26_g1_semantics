// Handling of CUDA errors
// Reference: Jason Sander and Edward Kandrot, CUDA by Example. Retrieved from https://developer.nvidia.com/cuda-example.
// Adjusted to meet our code format.
static void handle_error(cudaError_t error,
                         const char *file,
                         int line) {
  if (error != cudaSuccess) {
    std::cout << "[ERROR] " << cudaGetErrorString(error) << " in " << file << " at line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(error)(handle_error(error, __FILE__, __LINE__ ))
