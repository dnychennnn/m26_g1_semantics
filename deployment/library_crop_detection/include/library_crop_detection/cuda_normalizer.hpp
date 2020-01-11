#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_CUDA_NORMALIZER_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_CUDA_NORMALIZER_HPP_
/*!
 * @file cuda_normalizer.hpp
 *
 * @version 0.1
 */

namespace igg {

class CudaNormalizer() {
public:
  Normalize(*void buffer, std::vector<float> mean, std::vector<float> std);

      //for(int channel_index=0; channel_index<this->input_channels_; channel_index++) {
        //pixel[channel_index] = (pixel[channel_index]/255.0-this->mean_[channel_index])/this->std_[channel_index];
        //const int kBufferIndex = this->input_height_*this->input_width_*channel_index+position[0]*this->input_width_+position[1];
        //(static_cast<float*>(this->host_buffer_))[kBufferIndex] = pixel[channel_index];
      //}

private:
  int num_blocks_;
  int num_threads_;
};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_LIBRARY_CROP_DETECTION_CUDA_NORMALIZER_HPP_
