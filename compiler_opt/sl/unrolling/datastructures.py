import numpy as np
import dataclasses

# These need to be in a separate file (not __main__) to enable (un)pickling


@dataclasses.dataclass(frozen=True)
class UnrollDecisionRawSample:
    features: list
    base_runtime: np.array
    base_ci: np.array
    factor_runtimes: np.array
    factor_cis: np.array


@dataclasses.dataclass(frozen=True)
class UnrollDecisionTrainingSample:
    features: list
    advice: np.array
