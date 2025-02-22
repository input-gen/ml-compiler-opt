
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <optional>

namespace {
class Timer {
public:
  Timer() {
    if (auto c = getenv("MLGO_LOOP_UNROLL_PROFILING_FILE"))
      file = c;
  }
  ~Timer() {
    if (file && valid)
      std::ofstream(*file) << duration;
  }
  void begin() {
    if (timing)
      valid = false;
    timing = true;

    start_time = std::chrono::high_resolution_clock::now();
  }
  void end() {
    if (!timing)
      valid = false;
    timing = false;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto this_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                             end_time - start_time)
                             .count();
    duration += this_duration;
  }

private:
  std::optional<char *> file;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  uint64_t duration = 0;
  bool timing = false;
  bool valid = true;

} timer;
}

extern "C" void __mlgo_unrolled_loop_begin(void) { timer.begin(); }
extern "C" void __mlgo_unrolled_loop_end(void) { timer.end(); }
