# input-gen utilities

Baseline cmake configuration:

``` 
cmake $LLVM_PROJECT_ROOT/llvm -DCMAKE_ENABLE_PROJECTS="clang;lld" -DCMAKE_ENABLE_RUNTIMES="compiler-rt" -DCOMPILER_RT_BUILD_INPUTGEN=ON"
```

input-gen requires a C++20 standard library.

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
