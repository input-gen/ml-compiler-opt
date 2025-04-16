all: unrolling_profiler_test

CPU=

check: all
	./unrolling_profiler_test 1
	./unrolling_profiler_test 100
	./unrolling_profiler_test 10000

unrolling_profiler_test.o: unrolling_profiler_test.cpp
	clang++ -Wall -O3 unrolling_profiler_test.cpp -c -o unrolling_profiler_test.o -fPIE

unrolling_profiler.o: unrolling_profiler.cpp
	clang++ -Wall -O3 unrolling_profiler.cpp -c -o unrolling_profiler.o -fPIE -DCPU=$(CPU)

unrolling_profiler_test: unrolling_profiler_test.o unrolling_profiler.o
	clang++ -Wall -O3 unrolling_profiler_test.o unrolling_profiler.o -lpfm -o unrolling_profiler_test

clean:
	rm unrolling_profiler_test
	rm *.o
