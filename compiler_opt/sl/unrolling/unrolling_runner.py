
# coding=utf-8
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
"""Module for collect data of unroll runtime effect"""

import os
import random
import io
import sys
import argparse
import dataclasses
import json
import tempfile
import subprocess
import ctypes
import math
import pprint
from typing import Dict, Tuple, BinaryIO, Union, List, Optional

import gin
import tensorflow as tf

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import log_reader

from absl import logging

_DEFAULT_IDENTIFIER = 'default'

def send(f: BinaryIO, value: Union[int, float], spec: tf.TensorSpec):
    """Send the `value` - currently just a scalar - formatted as per `spec`."""

    if spec.dtype == tf.int64:
        convert_el_func = int
        ctype_func = ctypes.c_int64
    elif spec.dtype == tf.float32:
        convert_el_func = float
        ctype_func = ctypes.c_float
    else:
        print(spec.dtype, "not supported")
        assert False

    if isinstance(value, list):
        to_send = (ctype_func * len(value))(*[convert_el_func(el) for el in value])
    else:
        to_send = ctype_func(convert_el_func(value))

    assert f.write(bytes(to_send)) == ctypes.sizeof(ctype_func) * math.prod(
        spec.shape
    )
    f.flush()

# These need to be kept in sync with the ones in UnrollModelFeatureMaps.h
MAX_UNROLL_FACTOR = 32
UNROLL_FACTOR_OFFSET = 2
ADVICE_TENSOR_LEN = 1 + MAX_UNROLL_FACTOR - UNROLL_FACTOR_OFFSET

def make_response_for_factor(factor: int):
    l = [0.5 for _ in range(ADVICE_TENSOR_LEN)]
    if factor == 0 or factor == 1:
        return l
    assert(factor >= UNROLL_FACTOR_OFFSET)
    l[factor - UNROLL_FACTOR_OFFSET] = 2.0
    return l

@dataclasses.dataclass(frozen=True)
class UnrollDecisionResult:
    factor: int
    action: bool
    module: bytes

@dataclasses.dataclass(frozen=True)
class UnrollDecisionRuntime:
    factor: int
    runtime: Optional[float]

@dataclasses.dataclass(frozen=True)
class UnrollDecision:
    features: List
    results: List[UnrollDecisionResult]

class UnrollCompilerHost:
    def __init__(self):
        cur_decision = 0
        self.cur_action = None

        self.num_decisions = None
        self.decisions = None

        self.features = []

        self.tensor_mode = 'numpy'
        #self.tensor_mode = 'TensorValue'

        self.channel_base = None
        self.to_compiler = None
        self.from_compiler = None

    def on_features_collect(self, index, tensor_values):
        if self.tensor_mode == 'numpy':
            tensor_values = [tv.to_numpy() for tv in tensor_values]
        self.features.append(tensor_values)

    def on_heuristic_print(self, index, heuristic):
        logging.debug(heuristic)

    def on_action_print(self, index, action):
        logging.debug(action)

    def on_action_save(self, index, action):
        logging.debug(f'Saving action {action}')
        self.cur_action = action

    def get_replaced_response(self, heuristic, index, factor):
        return make_response_for_factor(heuristic)

    def read_heuristic(self, fc):
        event = json.loads(fc.readline())
        logging.debug(event)
        assert 'heuristic' in event
        heuristic = int.from_bytes(fc.read(8))
        fc.readline()
        return heuristic

    def read_action(self, fc):
        event = json.loads(fc.readline())
        logging.debug(event)
        assert 'action' in event
        action = bool(int.from_bytes(fc.read(1)))
        fc.readline()
        return action

    def get_unroll_decision_results(
            self,
            mod,
            working_dir: str):

        self.channel_base = os.path.join(working_dir, 'channel')
        self.to_compiler = self.channel_base + ".in"
        self.from_compiler = self.channel_base + ".out"
        try:
            logging.debug(f"Opening pipes {self.to_compiler} and {self.from_compiler}")
            os.mkfifo(self.to_compiler, 0o666)
            os.mkfifo(self.from_compiler, 0o666)

            self.num_decisions, _ = self.compile_once(
                mod,
                self.on_features_collect,
                self.on_heuristic_print,
                self.on_action_print,
                lambda index, tensor, heuristic: make_response_for_factor(heuristic)
               )

            logging.debug(f'Found {self.num_decisions} decisions to make')
            logging.debug(f'Collected features:{self.features}')

            for decision in range(self.num_decisions):
                decision_results = []
                # From factor = 1 (i.e. no unroll) to MAX_UNROLL_FACTOR inclusive
                for factor in range(1, MAX_UNROLL_FACTOR + 1):
                    _, out_module = self.compile_once(
                        mod,
                        lambda index, features: (),
                        lambda index, heuristic: (),
                        lambda index, action: self.on_action_save(index, action) if index == decision else self.on_action_print(index, action),
                        lambda index, tensor, heuristic: make_response_for_factor(factor) if index == decision else make_response_for_factor(heuristic)
                    )
                    decision_results.append(
                        UnrollDecisionResult(factor, self.cur_action, out_module))
                ud = UnrollDecision(self.features[decision], decision_results)
                logging.debug(pprint.pformat(ud))
                logging.debug('Got result:')
                yield ud
        finally:
            logging.debug(f"Closing pipes")
            os.unlink(self.to_compiler)
            os.unlink(self.from_compiler)

    def compile_once(
            self,
            mod,
            on_features,
            on_heuristic,
            on_action,
            get_response):

        cur_decision = 0

        # TODO we should specify the opt program or command line outside of this
        process_and_args = [
            'opt',
            '-O3',
            f'--mlgo-loop-unroll-interactive-channel-base={self.channel_base}',
            '--mlgo-loop-unroll-advisor-mode=development',
        ]

        logging.debug(f"Launching compiler {process_and_args}")
        compiler_proc = subprocess.Popen(
            process_and_args, stderr=subprocess.PIPE,
            # TODO we want to pipe our output module and return it
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE
        )
        logging.debug(f"Sending module")
        compiler_proc.stdin.write(mod)
        # FIXME is this the proper way to close the pipe? if we don't set it to
        # None then the communicate call will try to close it again and raise an
        # error
        compiler_proc.stdin.close()
        compiler_proc.stdin = None
        logging.debug(f"Starting communication")
        with io.BufferedWriter(io.FileIO(self.to_compiler, "wb")) as tc, \
             io.BufferedReader(io.FileIO(self.from_compiler, "rb")) as fc:
            header = log_reader._read_header(fc)
            tensor_specs = header.features
            advice_spec = header.advice
            context = None
            while compiler_proc.poll() is None:
                next_event = fc.readline()
                if not next_event:
                    break
                (
                    last_context,
                    observation_id,
                    features,
                    _,
                ) = log_reader.read_one_observation(
                    context, next_event, fc, tensor_specs, None
                )
                if last_context != context:
                    logging.debug(f"context: {last_context}")
                context = last_context
                logging.debug(f"observation: {observation_id}")
                tensor_values = []
                for fv in features:
                    logging.debug(fv.to_numpy())
                    tensor_values.append(fv)
                on_features(cur_decision, tensor_values)
                heuristic = self.read_heuristic(fc)
                on_heuristic(cur_decision, heuristic)
                send(tc, get_response(cur_decision, tensor_values, heuristic), advice_spec)
                on_action(cur_decision, self.read_action(fc))
                cur_decision += 1
        outs, errs = compiler_proc.communicate()
        logging.debug("Errs")
        logging.debug(errs.decode("utf-8"))
        logging.debug(f"Outs size {len(outs)}")
        compiler_proc.wait()

        return cur_decision, outs

@gin.configurable(module='runners')
class InliningRunner(compilation_runner.CompilationRunner):
  """Class for collecting data for inlining-for-size.

  Usage:
  inliner = InliningRunner(
                clang_path, llvm_size_path, launcher_path,
                moving_average_decay_rate)
  serialized_sequence_example, default_reward, moving_average_reward,
  policy_reward = inliner.collect_data(
      ir_path, tf_policy_path, default_reward, moving_average_reward)
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def compile_fn(
      self, mod,
      workdir: str) -> Dict[str, Tuple[tf.train.SequenceExample, float]]:
    ...

def get_module_runtime(module):
    return random.uniform(1.0, 2.0)

def get_udr_runtime(udr: UnrollDecisionResult):
    if udr.action or udr.factor == 1:
        return UnrollDecisionRuntime(udr.factor, get_module_runtime(udr.module))
    else:
        return UnrollDecisionRuntime(udr.factor, None)

def get_ud_sample(ud: UnrollDecision):
    x = ud.features
    y = [None for _ in range(ADVICE_TENSOR_LEN)]
    base_runtime = None
    for udr in ud.results:
        udrt = get_udr_runtime(udr)
        if udrt.factor == 1:
            base_runtime = udrt.runtime
        else:
            assert udrt.factor >= 2
            y[udrt.factor - UNROLL_FACTOR_OFFSET] = udrt.runtime

    # If none of the factors succeeded.
    if all(el is None for el in y):
        return None

    # Obtain speedup factors for all factors.
    # Encode failure to unroll as speedup of 0.0.
    y = [base_runtime / el if el is not None else 0.0 for el in y]

    return (x, y)


def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='Compiler host'
    )
    parser.add_argument('--module', required=True)
    parser.add_argument('--temp-dir', default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    main(args)

def main(args):

    if args.debug:
        logging.set_verbosity(logging.DEBUG)

    with open(args.module, 'rb') as f, \
         tempfile.TemporaryDirectory(dir=args.temp_dir) as tmpdir:
        mod = f.read()
        decision_results = UnrollCompilerHost().get_unroll_decision_results(mod, tmpdir)
        for ud in decision_results:
            sample = get_ud_sample(ud)
            logging.debug(f'Obtained sample {sample}')



if __name__ == "__main__":
    parse_args_and_run()
