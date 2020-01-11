#ifndef M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_STOP_WATCH_HPP_
#define M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_STOP_WATCH_HPP_
/*!
 * @file stop_watch.hpp
 *
 * Code provided by Marcell Missura.
 *
 * @version 0.1
 */

#include <time.h>

namespace igg {

class StopWatch {
public:
  StopWatch();
  ~StopWatch(){}
  void Start();
  double ElapsedTime();

private:
  struct timespec last_start_time_;
};

} // namespace igg

#endif // M26_G1_SEMANTICS_DEPLOYMENT_LIBRARY_CROP_DETECTION_INCLUDE_STOP_WATCH_HPP_
