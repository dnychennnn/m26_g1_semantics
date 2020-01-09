/*!
 * @file stop_watch.cpp
 *
 * Code provided by Marcell Missura.
 *
 * @version 0.1
 */
#include "stop_watch.hpp"

namespace igg {

StopWatch::StopWatch() {
  clock_gettime(CLOCK_MONOTONIC, &(this->last_start_time_));
}

// Resets the elapsed time to 0.
void StopWatch::Start() {
  clock_gettime(CLOCK_MONOTONIC, &last_start_time_);
}

// Returns a time stamp expressed in seconds since the last start.
double StopWatch::ElapsedTime() {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  long diff_in_secs = now.tv_sec - last_start_time_.tv_sec;
  long diff_in_nanos = now.tv_nsec - last_start_time_.tv_nsec;
  return static_cast<double>(diff_in_secs) + static_cast<double>(diff_in_nanos)/1000000000.0;
}

} // namespace igg
