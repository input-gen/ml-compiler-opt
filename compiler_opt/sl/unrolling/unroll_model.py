#!/usr/bin/env python3

# These need to be kept in sync with the ones in UnrollModelFeatureMaps.h
MAX_UNROLL_FACTOR = 32
UNROLL_FACTOR_OFFSET = 2
ADVICE_TENSOR_LEN = 1 + MAX_UNROLL_FACTOR - UNROLL_FACTOR_OFFSET
