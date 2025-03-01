#include <perfmon/perf_event.h>
#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_perf_event.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <cstring>

namespace {
class MLGOTimer {
public:
  MLGOTimer() {
    if (auto c = getenv("MLGO_LOOP_UNROLL_PROFILING_FILE"))
      file = c;

    const int initialize_pfm_result = pfm_initialize();
    if (initialize_pfm_result != PFM_SUCCESS) {
      std::cerr << pfm_strerror(initialize_pfm_result) << "\n";
      exit(1);
    }

    // Get the event encoding.
    char *fully_qualified_name = nullptr;
    pfm_perf_encode_arg_t get_encoding_arguments = {};
    perf_event_attr *attribute = new perf_event_attr();
    attribute->read_format =
        PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    get_encoding_arguments.attr = attribute;
    get_encoding_arguments.fstr = &fully_qualified_name;
    get_encoding_arguments.size = sizeof(pfm_perf_encode_arg_t);
    // unhalted_core_cycles is the name of the counter on most Intel CPUs to
    // count clock cycles. This needs to be adjusted for AMD CPUs. For newer
    // Zen based AMD uarches, it should be cycles_not_in_halt. Good to check
    // what the CyclesCounter value is for the uarch in question in LLVM's
    // x86PfmCounters.td file though.
    const int get_encoding_result =
        pfm_get_os_event_encoding("cycles_not_in_halt", PFM_PLM3,
                                  PFM_OS_PERF_EVENT, &get_encoding_arguments);
    if (fully_qualified_name) {
      free(fully_qualified_name);
    }
    if (get_encoding_result != PFM_SUCCESS) {
      std::cerr << pfm_strerror(get_encoding_result) << "\n";
      exit(1);
    }

    event_file_descriptor =
        perf_event_open(attribute, getpid(), -1, -1, 0);
    if (event_file_descriptor == -1) {
      std::cerr << "Failed to open event\n";
      exit(1);
    }
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

    ioctl(event_file_descriptor, PERF_EVENT_IOC_RESET);
  }
  void end() {
    if (!timing)
      valid = false;
    timing = false;

    ioctl(event_file_descriptor, PERF_EVENT_IOC_DISABLE);
    uint64_t event_info[3];
    ssize_t read_size = read(event_file_descriptor, &event_info, sizeof(event_info));
    if (read_size != sizeof(event_info)) {
      std::cerr << "Failed to read perf counter\n";
      exit(1);
    }

    duration += event_info[0];
  }

private:
  std::optional<char *> file;
  int event_file_descriptor = 0;
  uint64_t duration = 0;
  bool timing = false;
  bool valid = true;

} timer;
} // namespace

extern "C" void __mlgo_unrolled_loop_begin(void) { timer.begin(); }
extern "C" void __mlgo_unrolled_loop_end(void) { timer.end(); }
