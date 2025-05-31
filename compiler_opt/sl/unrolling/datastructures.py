import numpy as np
import dataclasses
from typing import List, Optional

# These need to be in a separate file (not __main__) to enable (un)pickling


@dataclasses.dataclass(frozen=True)
class UnrollDecisionTrainingSample:
    features: list
    heuristic_factor: int
    advice: np.array


@dataclasses.dataclass(frozen=True)
class AdaptiveBenchmarkingResult:
    runtimes: np.array
    mean: float
    ci: float
    converged: bool

    def is_zero_rt(self):
        return len(self.runtimes) == 2 and self.runtimes[0] == 0 and self.runtimes[1] == 0

    def is_invalid(self):
        return len(self.runtimes) == 0


def get_invalid_abr():
    return AdaptiveBenchmarkingResult(np.array([], dtype=float), np.nan, np.nan, False)


def get_zero_rt_abr():
    return AdaptiveBenchmarkingResult(
        np.array([float(0), float(0)], dtype=float), float(0), float(0), True
    )


@dataclasses.dataclass(frozen=True)
class UnrollFactorRuntimes:
    factor: int
    action: int
    # Benchmarking result for each input
    benchmarking_results: List[AdaptiveBenchmarkingResult]


@dataclasses.dataclass(frozen=True)
class UnrollDecisionRawSample:
    features: list
    # Benchmarking result for each input
    base_ufrts: UnrollFactorRuntimes
    # Benchmarking result for each input for each factor
    factors_ufrts: List[UnrollFactorRuntimes]
    # Benchmarking result for default decision
    default_ufrts: UnrollFactorRuntimes
