# input-gen utilities

## Python configuration
Version 3.11.0
``` shell
./configure --prefix=/path/to/install/dir --enable-shared --enable-loadable-sqlite-extensions --enable-optimizations
make -j
make install
```

## LLVM build configuration

Baseline cmake configuration for the llvm installation:

``` 
cmake $LLVM_PROJECT_ROOT/llvm -DCMAKE_ENABLE_PROJECTS="clang;lld" -DCMAKE_ENABLE_RUNTIMES="compiler-rt" -DCOMPILER_RT_BUILD_INPUTGEN=ON"
```

input-gen requires a C++20 conformant standard library.

## Scripts

The input-gen scripts generally accept the option `-mclang` to specify
additional flags to `clang` and `-mllvm` for additional options to `opt`.

Pass `--debug` to enable verbose output for debugging purposes.

`clang` is used for linking and compiling so depending on your environment,
additional flags may need to be specified.

For example to use a newer gcc toolchain rather than the default (required when
your default C++ std lib may be too old)
``` shell
-mclang='--gcc-toolchain=/path/to/gcc-toolchain' 
```

Multiple flags can also be specified as such:
``` shell
-mclang='--flag1' -mclang='--flag2' 
```

Note that this repo requires python `3.11`.

### Generating inputs for a module

``` shell
python3 input_gen.py --module input_module.ll (--entry-function=ENTRY_FUNCTION | --entry-all | --entry-marked)
```

### Generating ComPileLoop from ComPile

``` shell
PYTHONPATH=$PYTHONPATH:. python3 -m com_pile_utils.generate_com_pile_loop --dataset ~/datasets/ComPile --output-dataset ~/datasets/ComPileLoop.sqlite3
```

USR1 can be sent to get a status report, USR2 can be sent to abort and write out
the current pending database file.

### Generating ComPileLoop+Inputs from ComPileLoop

``` shell
python3 generate_com_pile_loop_inputs.py --dataset ~/datasets/ComPileLoop/ --output-dataset ./ComPileLoopInputs/
```

### Demo of how to process ComPileLoop

``` shell
python3 process_com_pile_loop_demo.py --dataset ~/datasets/ComPileLoop/
```

### Demo of how to process ComPileLoop+Inputs

``` shell
python3 process_com_pile_loop_inputs_demo.py --dataset ~/datasets/ComPileLoopInputs/
```

This will read the dataset and replay all inputs in it.

### Generating samples for training the unroll heuristic

Install the dependencies:

``` shell
dnf install libpfm-devel
```

Change to the root directory of this repo.

``` shell
cd compiler_opt/sl/unrolling/rts/
make CPU=AMD
# make CPU=INTEL
```

Check the compiled timing runtime:

```
make check
```

And go back to the root.

```
cd -
```

This should result in the following file:

``` shell
compiler_opt/sl/unrolling/rts/unrolling_profiler.o
```

The following can be used to generate training samples
``` shell
PYTHONPATH=$PYTHONPATH:. python3 -m compiler_opt.sl.unrolling.process_com_pile_loop_inputs --dataset ~/datasets/ComPileLoopInputs  --output-dataset ~/datasets//UnrollTrainingSamples.sqlite3 -mclang=compiler_opt/sl/unrolling/rts/unrolling_profiler.o  -mclang=-lpfm
```

Note the additional `-mclang` flag which links in the profiling runtime.

### Training the unroll heuristic

``` shell
PYTHONPATH=$PYTHONPATH:. python3 -m compiler_opt.sl.unrolling.train --dataset ~/datasets/UnrollTrainingSamples.sqlite3
```
