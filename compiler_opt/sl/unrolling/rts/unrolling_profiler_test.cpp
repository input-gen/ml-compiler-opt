#include <iostream>

extern "C" void __mlgo_unrolled_loop_begin(void);
extern "C" void __mlgo_unrolled_loop_end(void);

__attribute__((optnone))
void doWork() {
  int a;
  for (int i = 0; i < 100000; ++i) {
    a += 5;
  }
}

int main() {
  __mlgo_unrolled_loop_begin();
  doWork();
  __mlgo_unrolled_loop_end();
  return 0;
}
