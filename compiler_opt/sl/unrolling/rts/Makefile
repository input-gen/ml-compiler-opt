unrolling_profiler_test.o: unrolling_profiler_test.cpp
	clang++ -O3 unrolling_profiler_test.cpp -c -o unrolling_profiler_test.o

unrolling_profiler.o: unrolling_profiler.cpp
	clang++ -O3 unrolling_profiler.cpp -c -o unrolling_profiler.o

unrolling_profiler_test: unrolling_profiler_test.o unrolling_profiler.o
	clang++ -O3 unrolling_profiler_test.o unrolling_profiler.o -lpfm -o unrolling_profiler_test

clean:
	rm unrolling_profiler_test
	rm *.o
