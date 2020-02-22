/*
 * @file stem_extractor_gpu.cpp
 *
 * @version 0.1
 */

#include "library_crop_detection/stem_extractor_gpu.hpp"

#include <iostream>
#include <math.h>

#include <opencv2/core.hpp>
/*
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

__constant__ unsigned int constant_width;
__constant__ unsigned int constant_height;
__constant__ unsigned int constant_size;
__constant__ int constant_kernel_size_votes;
__constant__ int constant_kernel_size_peaks;
__constant__ float constant_keypoint_radius;
__constant__ float constant_threshold_votes;
__constant__ float constant_threshold_peaks;
__constant__ float constant_votes_normalization;

StemExtractorGpu::StemExtractorGpu(const StemExtractorParameters& kParameters):
    kKeypointRadius_{kParameters.keypoint_radius},
    kKernelSizeVotes_{kParameters.kernel_size_votes},
    kKernelSizePeaks_{kParameters.kernel_size_peaks},
    kThresholdVotes_{kParameters.threshold_votes},
    kThresholdPeaks_{kParameters.threshold_peaks},
    kVotesNormalization_{static_cast<float>(M_PI)*kParameters.keypoint_radius*kParameters.keypoint_radius}
{
}

StemExtractorGpu::~StemExtractorGpu() {
  this->FreeBuffers();
}

void StemExtractorGpu::FreeBuffers() {
  if (!this->is_loaded_) {return;}

  cudaFree(this->votes_device_);
  cudaFree(this->device_buffer_1_);
  cudaFree(this->device_buffer_2_);
  cudaFree(this->num_stem_positions_device_);
  cudaFreeHost(this->num_stem_positions_host_);
  cudaFree(this->stem_positions_device_);
  cudaFreeHost(this->stem_positions_host_);
}

void StemExtractorGpu::LoadAndAllocateBuffers(
    const unsigned int kHeight,
    const unsigned int kWidth,
    const float kScaling) {
  this->height_ = kHeight;
  this->width_ = kWidth;
  this->size_ = kHeight*kWidth;
  this->scaling_ = kScaling;

  cudaStream_t stream;
  HANDLE_ERROR(cudaStreamCreate(&(stream)));

  // allocate buffers

  this->FreeBuffers();

  HANDLE_ERROR(cudaMalloc(&(this->votes_device_), sizeof(float)*this->size_));
  HANDLE_ERROR(cudaMalloc(&(this->device_buffer_1_), sizeof(float)*this->size_));
  HANDLE_ERROR(cudaMalloc(&(this->device_buffer_2_), sizeof(float)*this->size_));
  HANDLE_ERROR(cudaMalloc(&(this->num_stem_positions_device_), sizeof(unsigned int)));
  HANDLE_ERROR(cudaMallocHost(&(this->num_stem_positions_host_), sizeof(unsigned int)));
  HANDLE_ERROR(cudaMalloc(&(this->stem_positions_device_), 3*sizeof(float)*this->size_)); // 3 is for x, y, score
  HANDLE_ERROR(cudaMallocHost(&(this->stem_positions_host_), 3*sizeof(float)*this->size_));

  // bring extraction parameters to constant GPU memory

  cudaMemcpyToSymbolAsync(constant_width, &(this->width_), sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_height, &(this->height_), sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_size, &(this->size_), sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_keypoint_radius, &(this->kKeypointRadius_), sizeof(float), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_threshold_votes, &(this->kThresholdVotes_), sizeof(float), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_threshold_peaks, &(this->kThresholdPeaks_), sizeof(float), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_kernel_size_votes, &(this->kKernelSizeVotes_), sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_kernel_size_peaks, &(this->kKernelSizePeaks_), sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(constant_votes_normalization, &(this->kVotesNormalization_), sizeof(float), 0, cudaMemcpyHostToDevice, stream);

  HANDLE_ERROR(cudaStreamSynchronize(stream));
  HANDLE_ERROR(cudaStreamDestroy(stream));

  this->is_loaded_ = true;
}

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
    index_y = index/constant_width;
    cast_y = index_y+static_cast<int>(offset_y);

    if (cast_y<0 || cast_y>=constant_height) {continue;}

    index_cast = cast_y*constant_width+cast_x;
    atomicAdd(&(votes[index_cast]), weight);
  }
}

__global__ void normalize(float* votes) {
  const size_t kInitialIndex = blockIdx.x*blockDim.x+threadIdx.x;
  const size_t kStride = blockDim.x*gridDim.x;

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
    float * input_1, float * input_2, float * positions, unsigned int * num_positions) {
  const size_t kInitialIndex = blockIdx.x*blockDim.x+threadIdx.x;
  const size_t kStride = blockDim.x*gridDim.x;

  unsigned int index_positions;

  for(int index = kInitialIndex; index<constant_size; index += kStride) {
    if (input_1[index]>=constant_threshold_peaks&&input_1[index]==input_2[index]) {
      index_positions = 3*atomicInc(num_positions, (unsigned int)(-1));

      positions[index_positions] = index%constant_width;
      positions[index_positions+1] = index/constant_width;
      positions[index_positions+2] = input_1[index];
    }
  }
}

/*
// get CUDA device attributes

int num_multi_processors;
HANDLE_ERROR(cudaDeviceGetAttribute(&num_multi_processors, cudaDevAttrMultiProcessorCount, kDeviceIndex));

int max_num_threads;
HANDLE_ERROR(cudaDeviceGetAttribute(&max_num_threads, cudaDevAttrMaxThreadsPerBlock, kDeviceIndex));
*/

template
void StemExtractorGpu::Infer(
    float* keypoint_confidence_device,
    float* keypoint_offsets_device,
    NetworkOutput& result) const;

template <const unsigned int kTileWidth,
          const unsigned int kTileHeight,
          const unsigned int kNumThreads,
          const unsigned int kNumBlocks,
          const unsigned int kDeviceIndex>
void StemExtractorGpu::Infer(
    float* keypoint_confidence_device,
    float* keypoint_offsets_device,
    NetworkOutput& result) const {

  if (!this->is_loaded_) {
    throw std::runtime_error(
        "StemExtractionGpu: Parameters not loaded and buffers not allocated. "
        "Did you call 'LoadAndAllocateBuffers()'?");
  }

  #ifdef DEBUG_MODE
  // measure extraction time in debug mode
  StopWatch stop_watch_total;
  StopWatch stop_watch;

  stop_watch_total.Start();
  #endif // DEBUG_MODE

  cudaStream_t stream;
  HANDLE_ERROR(cudaStreamCreate(&(stream)));

  // reset
  HANDLE_ERROR(cudaMemsetAsync(this->votes_device_, 0, this->size_*sizeof(float), stream));
  HANDLE_ERROR(cudaMemsetAsync(this->device_buffer_1_, 0, this->size_*sizeof(float), stream));
  HANDLE_ERROR(cudaMemsetAsync(this->device_buffer_2_, 0, this->size_*sizeof(float), stream));
  HANDLE_ERROR(cudaMemsetAsync(this->num_stem_positions_device_, 0, sizeof(unsigned int), stream));

  const unsigned int kNumThreadsY = kTileHeight;
  const unsigned int kNumThreadsX = kTileWidth;

  const unsigned int kNumBlocksY = (this->height_+kTileHeight-1)/kTileHeight;
  const unsigned int kNumBlocksX = (this->width_+kTileWidth-1)/kTileWidth;

  #ifdef DEBUG_MODE
  ROS_INFO("StemExtractionGpu: Using %d / %d, %d blocks and %d / %d, %d threads per block.",
      kNumBlocks, kNumBlocksX, kNumBlocksY, kNumThreads, kNumThreadsX, kNumThreadsY);
  #endif // DEBUG_MODE

  const dim3 kThreadDim(kNumThreadsX, kNumThreadsY);
  const dim3 kBlockDim(kNumBlocksX, kNumBlocksY);

  HANDLE_ERROR(cudaStreamSynchronize(stream));

  stop_watch.Start();

  cast_votes<<<kNumBlocks, kNumThreads, kDeviceIndex, stream>>>(
      keypoint_confidence_device,
      keypoint_offsets_device,
      this->votes_device_);

  HANDLE_ERROR(cudaStreamSynchronize(stream));

  normalize<<<kNumBlocks, kNumThreads, kDeviceIndex, stream>>>(this->votes_device_);

  HANDLE_ERROR(cudaStreamSynchronize(stream));

  #ifdef DEBUG_MODE
  ROS_INFO("Stem extraction time on GPU (cast votes): %f ms", 1000.0*stop_watch.ElapsedTime());
  stop_watch.Start();
  #endif // DEBUG_MODE

  box_filter<kTileWidth, kTileHeight><<<kBlockDim, kThreadDim, kDeviceIndex, stream>>>(
      this->votes_device_,
      this->device_buffer_1_);

  HANDLE_ERROR(cudaStreamSynchronize(stream));

  #ifdef DEBUG_MODE
  ROS_INFO("Stem extraction time on GPU (box filter): %f ms", 1000.0*stop_watch.ElapsedTime());
  stop_watch.Start();
  #endif // DEBUG_MODE

  max_filter<kTileWidth, kTileHeight><<<kBlockDim, kThreadDim, kDeviceIndex, stream>>>(
      this->device_buffer_1_,
      this->device_buffer_2_);

  HANDLE_ERROR(cudaStreamSynchronize(stream));

  #ifdef DEBUG_MODE
  ROS_INFO("Stem extraction time on GPU (max filter): %f ms", 1000.0*stop_watch.ElapsedTime());
  stop_watch.Start();
  #endif // DEBUG_MODE

  check_equal<<<kNumBlocks, kNumThreads, kDeviceIndex, stream>>>(
      this->device_buffer_1_,
      this->device_buffer_2_,
      this->stem_positions_device_,
      this->num_stem_positions_device_);

  HANDLE_ERROR(cudaStreamSynchronize(stream));

  #ifdef DEBUG_MODE
  ROS_INFO("Stem extraction time on GPU (find peaks): %f ms", 1000.0*stop_watch.ElapsedTime());
  stop_watch.Start();
  #endif // DEBUG_MODE

  HANDLE_ERROR(cudaMemcpyAsync(
      this->num_stem_positions_host_,
      this->num_stem_positions_device_,
      sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  HANDLE_ERROR(cudaMemcpyAsync(
      this->stem_positions_host_,
      this->stem_positions_device_,
      3*(*(this->num_stem_positions_host_))*sizeof(float), cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));

  std::vector<cv::Vec3f> stem_positions;
  stem_positions.reserve(*(this->num_stem_positions_host_));

  for (size_t position_index = 0; position_index<*(this->num_stem_positions_host_); position_index++) {
    stem_positions.emplace_back(cv::Vec3f(this->scaling_*this->stem_positions_host_[3*position_index],
                                          this->scaling_*this->stem_positions_host_[3*position_index+1],
                                          this->stem_positions_host_[3*position_index+2]));
  }

  result.SetStemPositions(std::move(stem_positions));

  #ifdef DEBUG_MODE
  ROS_INFO("Stem extraction time on GPU (copy result): %f ms", 1000.0*stop_watch.ElapsedTime());
  #endif // DEBUG_MODE

  #ifdef DEBUG_MODE
  double extraction_time = stop_watch_total.ElapsedTime();
  ROS_INFO("Stem extraction time on GPU (total): %f ms (%f fps)", 1000.0*extraction_time, 1.0/extraction_time);
  #endif // DEBUG_MODE

  #ifdef DEBUG_MODE
  // copy votes for visualization
  void* votes_host = result.ServeVotesBuffer(this->height_, this->width_);
  HANDLE_ERROR(cudaMemcpyAsync(votes_host, this->votes_device_, this->size_*sizeof(float), cudaMemcpyDeviceToHost, stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  #endif // DEBUG_MODE

  HANDLE_ERROR(cudaStreamDestroy(stream));
}

} // namespace igg
