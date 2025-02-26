
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>

namespace {
class MLGOTimer {
public:
  MLGOTimer() {
    if (auto c = getenv("MLGO_LOOP_UNROLL_PROFILING_FILE"))
      file = c;
  }
  ~MLGOTimer() {
    std::ofstream ofs;
    std::ostream *os;
    if (file) {
      ofs.open(*file);
      os = &ofs;
    } else {
        os = &std::cout;
    }
    if (valid)
      *os << "MLGO_LOOP_UNROLL_TIMER " << duration << "\n";
    else
      *os << "MLGO_LOOP_UNROLL_TIMER_INVALID\n";
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
    auto this_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
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
