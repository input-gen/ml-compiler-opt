#include <cstdlib>

extern "C" void __mlgo_unrolled_loop_begin(void);
extern "C" void __mlgo_unrolled_loop_end(void);

__attribute__((optnone)) void doWork(int n) {
  [[maybe_unused]] int a = 0;
  for (int i = 0; i < n; ++i) {
    a += 5;
  }
}

int main(int argc, char **argv) {
  __mlgo_unrolled_loop_begin();
  doWork(atoi(argv[1]));
  __mlgo_unrolled_loop_end();
  return 0;
}
