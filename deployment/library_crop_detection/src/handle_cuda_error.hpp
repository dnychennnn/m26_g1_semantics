#ifdef TENSORRT_AVAILABLE
// Handling of CUDA errors
// Reference: Jason Sander and Edward Kandrot, CUDA by Example. Retrieved from https://developer.nvidia.com/cuda-example.
// Adjusted to meet our code format.
static void handle_cuda_error(const cudaError_t kError, const char* kFile, const int kLine) {
  if (kError!=cudaSuccess) {
    std::cerr << cudaGetErrorString(kError) << " in " << kFile << " at line " << kLine << "\n";
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(error)(handle_cuda_error(error, __FILE__, __LINE__ ))
#endif // TENSORRT_AVAILABLE
