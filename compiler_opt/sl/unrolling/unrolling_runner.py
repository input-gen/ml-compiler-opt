
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
from typing import Dict, Tuple, BinaryIO, Union, List

import gin
import tensorflow as tf

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import log_reader

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
    runtime: float

@dataclasses.dataclass(frozen=True)
class UnrollDecision:
    features: List
    results: List[UnrollDecisionResult]

class UnrollCompilerHost:
    def __init__(self):
        self.cur_decision = 0
        self.cur_action = None

        self.num_decisions = None
        self.decisions = None

        self.features = []

        self.tensor_mode = 'numpy'
        #self.tensor_mode = 'TensorValue'

    def on_features_collect(self, index, tensor_values):
        if self.tensor_mode == 'numpy':
            tensor_values = [tv.to_numpy() for tv in tensor_values]
        self.features.append(tensor_values)

    def on_heuristic_print(self, index, heuristic):
        print(heuristic)

    def on_action_print(self, index, action):
        print(action)

    def on_action_save(self, index, action):
        print(f'Saving action {action}')
        self.cur_action = action

    def get_replaced_response(self, heuristic, index, factor):
        return make_response_for_factor(heuristic)

    def read_heuristic(self, fc):
        event = json.loads(fc.readline())
        print(event)
        assert 'heuristic' in event
        heuristic = int.from_bytes(fc.read(8))
        fc.readline()
        return heuristic

    def read_action(self, fc):
        event = json.loads(fc.readline())
        print(event)
        assert 'action' in event
        action = bool(int.from_bytes(fc.read(1)))
        fc.readline()
        return action

    def get_runtime(self, module):
        return None

    def handle_module(
            self,
            mod,
            working_dir: str):
        self.compile_once(
            mod, working_dir,
            self.on_features_collect,
            self.on_heuristic_print,
            self.on_action_print,
            lambda index, tensor, heuristic: make_response_for_factor(heuristic)
        )
        self.num_decisions = self.cur_decision

        print(f'Found {self.num_decisions} decisions to make')
        print(f'Collected features:{self.features}')

        results = []

        for decision in range(self.num_decisions):
            decision_results = []
            # From factor = 1 (i.e. no unroll) to MAX_UNROLL_FACTOR inclusive
            for factor in range(1, MAX_UNROLL_FACTOR + 1):
                out_module = self.compile_once(
                    mod, working_dir,
                    lambda index, features: (),
                    lambda index, heuristic: (),
                    lambda index, action: self.on_action_save(index, action) if index == decision else self.on_action_print(index, action),
                    lambda index, tensor, heuristic: make_response_for_factor(factor) if index == decision else make_response_for_factor(heuristic)
                )
                decision_results.append(
                    UnrollDecisionResult(factor, self.cur_action, self.get_runtime(out_module)))
            results.append(UnrollDecision(self.features[decision], decision_results))


        print('Got results:')
        pprint.pp(results)
        # print(*results, sep='\n')

    def compile_once(
            self,
            mod,
            working_dir: str,
            on_features,
            on_heuristic,
            on_action,
            get_response):

        self.cur_decision = 0

        temp_rootname = os.path.join(working_dir, 'channel')

        # TODO we should specify the opt program or command line outside of this
        process_and_args = [
            'opt',
            '-O3',
            f'--mlgo-loop-unroll-interactive-channel-base={temp_rootname}',
            '--mlgo-loop-unroll-advisor-mode=development',
        ]

        to_compiler = temp_rootname + ".in"
        from_compiler = temp_rootname + ".out"
        print(f"Opening pipes {to_compiler} and {from_compiler}")
        try:
            os.mkfifo(to_compiler, 0o666)
            os.mkfifo(from_compiler, 0o666)
            print(f"Launching compiler {process_and_args}")
            compiler_proc = subprocess.Popen(
                process_and_args, stderr=subprocess.PIPE,
                # TODO we want to pipe our output module and return it
                stdout=subprocess.DEVNULL,
                stdin=subprocess.PIPE
            )
            print(f"Sending module")
            compiler_proc.stdin.write(mod)
            # FIXME is this the proper way to close the pipe? if we don't set it to
            # None then the communicate call will try to close it again and raise an
            # error
            compiler_proc.stdin.close()
            compiler_proc.stdin = None
            print(f"Starting communication")
            with io.BufferedWriter(io.FileIO(to_compiler, "wb")) as tc, \
                 io.BufferedReader(io.FileIO(from_compiler, "rb")) as fc:
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
                        print(f"context: {last_context}")
                    context = last_context
                    print(f"observation: {observation_id}")
                    tensor_values = []
                    for fv in features:
                        print(fv.to_numpy())
                        tensor_values.append(fv)
                    on_features(self.cur_decision, tensor_values)
                    heuristic = self.read_heuristic(fc)
                    on_heuristic(self.cur_decision, heuristic)
                    send(tc, get_response(self.cur_decision, tensor_values, heuristic), advice_spec)
                    on_action(self.cur_decision, self.read_action(fc))
                    self.cur_decision += 1
            _, err = compiler_proc.communicate()
            print(err.decode("utf-8"))
            compiler_proc.wait()

        finally:
            os.unlink(to_compiler)
            os.unlink(from_compiler)

        return None

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


def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='Compiler host'
    )
    parser.add_argument('--module', required=True)
    parser.add_argument('--temp-dir', default=None)
    args = parser.parse_args()
    main(args)

def main(args):
    with open(args.module, 'rb') as f, \
         tempfile.TemporaryDirectory(dir=args.temp_dir) as tmpdir:
        mod = f.read()
        UnrollCompilerHost().handle_module(mod, tmpdir)


if __name__ == "__main__":
    parse_args_and_run()
