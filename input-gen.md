# Document Title


## Generating ComPileLoop from ComPile
```
python3 ./llvm/utils/mlgo-utils/com-pile-utils/extract_loops.py --dataset path/to/ComPile-Public-V2/ --output-dataset ./ComPileLoop --end 100
```

USR1 can be sent to get a status report, USR2 can be sent to abort and write out the current pending database file.

## Hosting a compiler making unroll decisions

Running this file demoes how to obtain training samples for a module.
```
PYTHONPATH=$PYTHONPATH:. python3 compiler_opt/sl/unrolling/unrolling_runner.py --module path/to/llvm-project/llvm/test/Transforms/LoopUnroll/ML/Inputs/nested.ll  --debug -S 
```

Currently just returns random numbers for the training samples - needs to be hooked up with input-gen.







