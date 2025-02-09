
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
import json
import tempfile
import subprocess
import ctypes
import math
from typing import Dict, Tuple, BinaryIO, Union

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

MAX_UNROLL_FACTOR = 32
UNROLL_FACTOR_OFFSET = 2

ADVICE_TENSOR_LEN = MAX_UNROLL_FACTOR - UNROLL_FACTOR_OFFSET

def make_response_for_factor(factor: int):
    l = [0.5 for _ in range(ADVICE_TENSOR_LEN)]
    if factor == 0 or factor == 1:
        return l
    assert(factor >= UNROLL_FACTOR_OFFSET)
    l[factor - UNROLL_FACTOR_OFFSET] = 2.0
    return l


class UnrollCompilerHost:
    def __init__(self):
        self.num_decisions = None

    def read_heuristic(self, fc):
        event = json.loads(fc.readline())
        print(event)
        assert 'heuristic' in event
        heuristic = int.from_bytes(fc.read(8))
        print(heuristic)
        fc.readline()
        return heuristic

    def read_action(self, fc):
        event = json.loads(fc.readline())
        print(event)
        assert 'action' in event
        action = bool(int.from_bytes(fc.read(1)))
        print(action)
        fc.readline()
        return action

    def handle_module(
            self,
            mod,
            working_dir: str):

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
                    heuristic = self.read_heuristic(fc)
                    send(tc, make_response_for_factor(heuristic), advice_spec)
                    action = self.read_action(fc)
            _, err = compiler_proc.communicate()
            print(err.decode("utf-8"))
            compiler_proc.wait()

        finally:
            os.unlink(to_compiler)
            os.unlink(from_compiler)

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
