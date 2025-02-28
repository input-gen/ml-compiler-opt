# input-gen utilities

Baseline cmake configuration for the llvm installation:

``` 
cmake $LLVM_PROJECT_ROOT/llvm -DCMAKE_ENABLE_PROJECTS="clang;lld" -DCMAKE_ENABLE_RUNTIMES="compiler-rt" -DCOMPILER_RT_BUILD_INPUTGEN=ON"
```

input-gen requires a C++20 standard library.

The input-gen scripts generally accept the option `-mclang` to specify
additional flags to `clang` and `-mllvm` for additional options to `opt`.

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

Note that the scripts in this directory generally require python `3.12` or
newer, while the scripts outside of this require python `3.11`.

## Generating ComPileLoop from ComPile

``` shell
python3 generate_com_pile_loop.py --dataset path/to/ComPile/ --output-dataset ./ComPileLoop --output-dataset-json ./ComPileLoopJson --end 100
```

USR1 can be sent to get a status report, USR2 can be sent to abort and write out
the current pending database file.

The script also generates JSON files containing a summary of each generated
parquet file. This information can be used to continue generation of the
database from where the process was interrupted. 

## Generating inputs for a module

``` shell
python3 input_gen.py --module input_module.ll {--entry-all,--entry-function=foo,--entry-marked}
```

## Generating ComPileLoop+Inputs from ComPileLoop

``` shell
python3 generate_com_pile_loop_inputs.py --dataset path/to/ComPileLoop/ --output-dataset ./ComPileLoopInputs/ --output-dataset-json ./ComPileLoopInputsJson
```

## Demo of how to process ComPileLoop

``` shell
python3 process_com_pile_loop_demo.py --dataset path/to/ComPileLoop/
```

## Demo of how to process ComPileLoop+Inputs

``` shell
python3 process_com_pile_loop_inputs_demo.py --dataset path/to/ComPileLoopInputs/
```

This will read the dataset and replay all inputs in it.

## Generating samples for training the unroll heuristic

First change to the root directory of this repo.

Prepare the runtime for collecting timing information:
``` shell
clang++ -c -emit-llvm -O2 compiler_opt/sl/unrolling/rts/unrolling_profiler.cpp -o compiler_opt/sl/unrolling/rts/unrolling_profiler.bc
```

The following can be used to generate training samples
``` shell
PYTHONPATH=$PYTHONPATH:. python3 compiler_opt/sl/unrolling/process_com_pile_loop_inputs.py --dataset path/to/ComPileLoopInputs -mclang=compiler_opt/sl/unrolling/rts/unrolling_profiler.bc
```

Note the additional `-mclang` flag which links in the profiling runtime.

