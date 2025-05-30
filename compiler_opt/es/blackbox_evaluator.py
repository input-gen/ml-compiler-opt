# coding=utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tooling computing rewards in blackbox learner."""

import abc
import concurrent.futures
from typing import List, Optional

from absl import logging
import gin

from compiler_opt.distributed.worker import FixedWorkerPool
from compiler_opt.rl import corpus
from compiler_opt.es import blackbox_optimizers
from compiler_opt.distributed import buffered_scheduler
from compiler_opt.rl import policy_saver


class BlackboxEvaluator(metaclass=abc.ABCMeta):
  """Blockbox evaluator abstraction."""

  @abc.abstractmethod
  def __init__(self, train_corpus: corpus.Corpus):
    pass

  @abc.abstractmethod
  def get_results(
      self, pool: FixedWorkerPool, perturbations: List[policy_saver.Policy]
  ) -> List[concurrent.futures.Future]:
    raise NotImplementedError()

  @abc.abstractmethod
  def set_baseline(self, pool: FixedWorkerPool) -> None:
    raise NotImplementedError()

  def get_rewards(
      self, results: List[concurrent.futures.Future]) -> List[Optional[float]]:
    rewards = [None] * len(results)

    for i in range(len(results)):
      if not results[i].exception():
        rewards[i] = results[i].result()
      else:
        logging.info('Error retrieving result from future: %s',
                     str(results[i].exception()))

    return rewards


@gin.configurable
class SamplingBlackboxEvaluator(BlackboxEvaluator):
  """A blackbox evaluator that samples from a corpus to collect reward."""

  def __init__(self, train_corpus: corpus.Corpus,
               estimator_type: blackbox_optimizers.EstimatorType,
               total_num_perturbations: int, num_ir_repeats_within_worker: int):
    self._samples = []
    self._train_corpus = train_corpus
    self._total_num_perturbations = total_num_perturbations
    self._num_ir_repeats_within_worker = num_ir_repeats_within_worker
    self._estimator_type = estimator_type

    super().__init__(train_corpus)

  def get_results(
      self, pool: FixedWorkerPool, perturbations: List[policy_saver.Policy]
  ) -> List[concurrent.futures.Future]:
    if not self._samples:
      for _ in range(self._total_num_perturbations):
        sample = self._train_corpus.sample(self._num_ir_repeats_within_worker)
        self._samples.append(sample)
        # add copy of sample for antithetic perturbation pair
        if self._estimator_type == (
            blackbox_optimizers.EstimatorType.ANTITHETIC):
          self._samples.append(sample)

    compile_args = zip(perturbations, self._samples)

    _, futures = buffered_scheduler.schedule_on_worker_pool(
        action=lambda w, v: w.compile(v[0], v[1]),
        jobs=compile_args,
        worker_pool=pool)

    not_done = futures
    # wait for all futures to finish
    while not_done:
      # update lists as work gets done
      _, not_done = concurrent.futures.wait(
          not_done, return_when=concurrent.futures.FIRST_COMPLETED)

    return futures

  def set_baseline(self, pool: FixedWorkerPool) -> None:
    del pool  # Unused.
    pass


@gin.configurable
class TraceBlackboxEvaluator(BlackboxEvaluator):
  """A blackbox evaluator that utilizes trace based cost modelling."""

  def __init__(self, train_corpus: corpus.Corpus,
               estimator_type: blackbox_optimizers.EstimatorType,
               bb_trace_path: str, function_index_path: str):
    self._train_corpus = train_corpus
    self._estimator_type = estimator_type
    self._bb_trace_path = bb_trace_path
    self._function_index_path = function_index_path

    self._baseline: Optional[float] = None

  def get_results(
      self, pool: FixedWorkerPool, perturbations: List[policy_saver.Policy]
  ) -> List[concurrent.futures.Future]:
    job_args = [{
        'modules': self._train_corpus.module_specs,
        'function_index_path': self._function_index_path,
        'bb_trace_path': self._bb_trace_path,
        'tflite_policy': perturbation
    } for perturbation in perturbations]

    _, futures = buffered_scheduler.schedule_on_worker_pool(
        action=lambda w, args: w.compile_corpus_and_evaluate(**args),
        jobs=job_args,
        worker_pool=pool)
    concurrent.futures.wait(
        futures, return_when=concurrent.futures.ALL_COMPLETED)
    return futures

  def set_baseline(self, pool: FixedWorkerPool) -> None:
    if self._baseline is not None:
      raise RuntimeError('The baseline has already been set.')

    job_args = [{
        'modules': self._train_corpus.module_specs,
        'function_index_path': self._function_index_path,
        'bb_trace_path': self._bb_trace_path,
        'tflite_policy': None,
    }]

    _, futures = buffered_scheduler.schedule_on_worker_pool(
        action=lambda w, args: w.compile_corpus_and_evaluate(**args),
        jobs=job_args,
        worker_pool=pool)

    concurrent.futures.wait(
        futures, return_when=concurrent.futures.ALL_COMPLETED)
    if len(futures) != 1:
      raise ValueError('Expected to have one result for setting the baseline,'
                       f' got {len(futures)}')

    self._baseline = futures[0].result()
