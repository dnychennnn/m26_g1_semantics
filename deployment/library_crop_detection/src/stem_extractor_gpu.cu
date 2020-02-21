/*
 * @file stem_extractor_gpu.cpp
 *
 * @version 0.1
 */

#include "library_crop_detection/stem_extractor_gpu.hpp"

#include <iostream>
#include <math.h>

/*
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
*/

#ifdef DEBUG_MODE
#include <ros/console.h>
#include "library_crop_detection/stop_watch.hpp"
#endif // DEBUG_MODE

#include "handle_cuda_error.hpp"


namespace igg {

StemExtractorGpu::StemExtractorGpu(const StemExtractorParameters& kParameters):
    kKeypointRadius_{kParameters.keypoint_radius},
    kKernelSizeVotes_{kParameters.kernel_size_votes},
    kKernelSizePeaks_{kParameters.kernel_size_peaks},
    kThresholdVotes_{kParameters.threshold_votes},
    kThresholdPeaks_{kParameters.threshold_peaks},
    kVotesNormalization_{static_cast<float>(M_PI)*kParameters.keypoint_radius*kParameters.keypoint_radius}
{}

StemExtractorGpu::~StemExtractorGpu() {}

__constant__ int constant_width;
__constant__ int constant_height;
__constant__ int constant_size;
__constant__ float constant_keypoint_radius;
__constant__ float constant_threshold_votes;
__constant__ float constant_threshold_peaks;
__constant__ int constant_kernel_size_votes;
__constant__ int constant_kernel_size_peaks;
__constant__ float constant_votes_normalization;

__global__ void cast_votes(
    float * keypoint_confidence, float * keypoint_offsets, float * votes) {

  const size_t kInitialIndex = blockIdx.x*blockDim.x+threadIdx.x;
  const size_t kStride = blockDim.x*gridDim.x;

  int index_x, index_y, index_cast, cast_x, cast_y;
  float offset_x, offset_y, weight;

  for(int index = kInitialIndex; index<constant_size; index += kStride) {
    weight = keypoint_confidence[index];

    if (weight<constant_threshold_votes) {continue;}

    offset_x = constant_keypoint_radius*keypoint_offsets[index];
    index_x = index%constant_width;
    cast_x = index_x+static_cast<int>(offset_x);

    if (cast_x<0 || cast_x>=constant_width) {continue;}

    offset_y = constant_keypoint_radius*keypoint_offsets[constant_size+index];
    index_y = index/constant_height;
    cast_y = index_y+static_cast<int>(offset_y);

    if (cast_y<0 || cast_y>=constant_height) {continue;}

    index_cast = cast_y*constant_width+cast_x;
    atomicAdd(&(votes[index_cast]), weight);
  }

  __syncthreads();

  // normalize
  for(int index = kInitialIndex; index<constant_size; index += kStride) {
    votes[index] /= constant_votes_normalization;
  }
}


template <const unsigned int kTileWidth, const unsigned int kTileHeight>
__global__ void box_filter(float* input, float* output) {

  __shared__ float shared_tile[kTileHeight][kTileWidth];

  const int kIndexX = blockIdx.x*blockDim.x+threadIdx.x;
  if (kIndexX<0 || kIndexX>=constant_width) {return;}

  const int kIndexY = blockIdx.y*blockDim.y+threadIdx.y;
  if (kIndexY<0 || kIndexY>=constant_height) {return;}

  const int kIndex = kIndexY*constant_width+kIndexX;

  // put tile into shared memory
  if (kIndexX>=0 && kIndexX<constant_width && kIndexY>=0 && kIndexY<constant_height) {
    shared_tile[threadIdx.y][threadIdx.x] = input[kIndex];
  }

  int tile_index_x, tile_index_y, index_x, index_y;

  __syncthreads();

  for (int offset_y = -constant_kernel_size_votes/2; offset_y<=constant_kernel_size_votes/2; offset_y++) {
    for (int offset_x = -constant_kernel_size_votes/2; offset_x<=constant_kernel_size_votes/2; offset_x++) {
      index_x = kIndexX+offset_x;
      if (index_x<0 || index_x>=constant_width) {continue;}

      index_y = kIndexY+offset_y;
      if (index_y<0 || index_y>=constant_height) {continue;}

      tile_index_x = threadIdx.x+offset_x;
      tile_index_y = threadIdx.y+offset_y;

      if (tile_index_x>=0 && tile_index_x<kTileWidth
          && tile_index_y>=0 && tile_index_y<kTileHeight) {
        output[kIndex] += shared_tile[tile_index_y][tile_index_x];
      } else {
        output[kIndex] += input[index_y*constant_width+index_x];
      }
    }
  }
}


template <const unsigned int kTileWidth, const unsigned int kTileHeight>
__global__ void max_filter(float* input, float* output) {

  __shared__ float shared_tile[kTileHeight][kTileWidth];

  const int kIndexX = blockIdx.x*blockDim.x+threadIdx.x;
  if (kIndexX<0 || kIndexX>=constant_width) {return;}

  const int kIndexY = blockIdx.y*blockDim.y+threadIdx.y;
  if (kIndexY<0 || kIndexY>=constant_height) {return;}

  const int kIndex = kIndexY*constant_width+kIndexX;

  // put tile into shared memory
  if (kIndexX>=0 && kIndexX<constant_width && kIndexY>=0 && kIndexY<constant_height) {
    shared_tile[threadIdx.y][threadIdx.x] = input[kIndex];
  }

  int tile_index_x, tile_index_y, index_x, index_y;
  float maximum = 0.0; // assuming non-negative values

  __syncthreads();

  for (int offset_y = -constant_kernel_size_votes/2; offset_y<=constant_kernel_size_votes/2; offset_y++) {
    for (int offset_x = -constant_kernel_size_votes/2; offset_x<=constant_kernel_size_votes/2; offset_x++) {
      index_x = kIndexX+offset_x;
      if (index_x<0 || index_x>=constant_width) {continue;}

      index_y = kIndexY+offset_y;
      if (index_y<0 || index_y>=constant_height) {continue;}

      tile_index_x = threadIdx.x+offset_x;
      tile_index_y = threadIdx.y+offset_y;

      if (tile_index_x>=0 && tile_index_x<kTileWidth
          && tile_index_y>=0 && tile_index_y<kTileHeight) {
        if (maximum<shared_tile[tile_index_y][tile_index_x]) {
          maximum = shared_tile[tile_index_y][tile_index_x];
        }
      } else {
        output[kIndex] += input[index_y*constant_width+index_x];
        if (maximum<input[index_y*constant_width+index_x]) {
          maximum = input[index_y*constant_width+index_x];
        }
      }
    }
  }

  output[kIndex] = maximum;
}


__global__ void check_equal(
    float * input_1, float * input_2) {
  const size_t kInitialIndex = blockIdx.x*blockDim.x+threadIdx.x;
  const size_t kStride = blockDim.x*gridDim.x;

  for(int index = kInitialIndex; index<constant_size; index += kStride) {
    input_1[index] = input_1[index]>=constant_threshold_peaks&&input_1[index]==input_2[index] ? input_1[index] : 0.0f;
  }
}


void StemExtractorGpu::Infer(void * keypoint_confidence, void * keypoint_offsets,
      const int kHeight, const int kWidth, NetworkOutput& result) const {

  #ifdef DEBUG_MODE
  // measure extraction time in debug mode
  StopWatch stop_watch;
  #endif // DEBUG_MODE

  const int kSize = kWidth*kHeight;
  const int kDeviceIndex = 0; // always use first CUDA device

  // use a separate cuda stream
  cudaStream_t stream;
  HANDLE_ERROR(cudaStreamCreate(&stream));

  int num_multi_processors;
  HANDLE_ERROR(cudaDeviceGetAttribute(&num_multi_processors, cudaDevAttrMultiProcessorCount, kDeviceIndex));

  int max_num_threads;
  HANDLE_ERROR(cudaDeviceGetAttribute(&max_num_threads, cudaDevAttrMaxThreadsPerBlock, kDeviceIndex));

  const int kNumBlocks = 8*num_multi_processors;
  const int kNumThreads = max_num_threads;

  void* votes_device;
  HANDLE_ERROR(cudaMalloc(&votes_device, 4*kSize)); // 4 bytes per float
  HANDLE_ERROR(cudaMemsetAsync(votes_device, 0, 4*kSize, stream));

  void* device_buffer_1;
  HANDLE_ERROR(cudaMalloc(&device_buffer_1, 4*kSize)); // 4 bytes per float
  HANDLE_ERROR(cudaMemsetAsync(device_buffer_1, 0, 4*kSize, stream));

  void* device_buffer_2;
  HANDLE_ERROR(cudaMalloc(&device_buffer_2, 4*kSize)); // 4 bytes per float
  HANDLE_ERROR(cudaMemsetAsync(device_buffer_2, 0, 4*kSize, stream));

  void* votes_result = result.ServeVotesBuffer(kHeight, kWidth);

  const unsigned int kTileHeight = 32;
  const unsigned int kTileWidth = 32;

  const unsigned int kNumThreadsY = kTileHeight;
  const unsigned int kNumThreadsX = kTileWidth;

  const unsigned int kNumBlocksY = (kHeight+kTileHeight-1)/kTileHeight;
  const unsigned int kNumBlocksX = (kWidth+kTileWidth-1)/kTileWidth;

  ROS_INFO("Using (%d, %d) blocks and (%d, %d) threads per block.", kNumBlocksX, kNumBlocksY, kNumThreadsX, kNumThreadsY);

  const dim3 kThreadDim(kNumThreadsX, kNumThreadsY);
  const dim3 kBlockDim(kNumBlocksX, kNumBlocksY);

  cudaMemcpyToSymbolAsync(constant_width, &kWidth, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_height, &kWidth, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_size, &kSize, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_keypoint_radius, &(this->kKeypointRadius_), sizeof(float), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_threshold_votes, &(this->kThresholdVotes_), sizeof(float), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_threshold_peaks, &(this->kThresholdPeaks_), sizeof(float), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_kernel_size_votes, &(this->kKernelSizeVotes_), sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_kernel_size_peaks, &(this->kKernelSizePeaks_), sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_votes_normalization, &(this->kVotesNormalization_), sizeof(float), 0, cudaMemcpyHostToDevice, stream);

  #ifdef DEBUG_MODE
  ROS_INFO("Using %d blocks and %d threads per block for stem extraction.", kNumBlocks, kNumThreads);
  #endif // DEBUG_MODE

  HANDLE_ERROR(cudaStreamSynchronize(stream));

  cast_votes<<<kNumBlocks, kNumThreads, kDeviceIndex, stream>>>(
      static_cast<float*>(keypoint_confidence),
      static_cast<float*>(keypoint_offsets),
      static_cast<float*>(votes_device));

  box_filter<kTileWidth, kTileHeight><<<kBlockDim, kThreadDim, kDeviceIndex, stream>>>(
      static_cast<float*>(votes_device),
      static_cast<float*>(device_buffer_1));

  max_filter<kTileWidth, kTileHeight><<<kBlockDim, kThreadDim, kDeviceIndex, stream>>>(
      static_cast<float*>(device_buffer_1),
      static_cast<float*>(device_buffer_2));

  check_equal<<<kNumBlocks, kNumThreads, kDeviceIndex, stream>>>(
      static_cast<float*>(device_buffer_1),
      static_cast<float*>(device_buffer_2));

  HANDLE_ERROR(cudaStreamSynchronize(stream));

  HANDLE_ERROR(cudaMemcpyAsync(votes_result, device_buffer_1, 4*kSize, cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  HANDLE_ERROR(cudaStreamDestroy(stream));

  #ifdef DEBUG_MODE
  double extraction_time = stop_watch.ElapsedTime();
  ROS_INFO("Stem extraction time (GPU): %f ms (%f fps)", 1000.0*extraction_time, 1.0/extraction_time);
  #endif // DEBUG_MODE
}

} // namespace igg
