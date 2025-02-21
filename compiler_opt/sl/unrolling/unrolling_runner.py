
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
import fcntl
from typing import Dict, Tuple, BinaryIO, Union, List, Optional, Iterable

import gin
import tensorflow as tf

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import log_reader

from absl import logging

_DEFAULT_IDENTIFIER = 'default'

def send_instrument_response(f: BinaryIO, response: Optional[Tuple[str, str]]):
    if response is None:
        f.write(bytes([0]))
        f.flush()
    else:
        f.write(bytes([1]))
        begin = response[0].encode('ascii') + bytes([0])
        end = response[1].encode('ascii') + bytes([0])
        f.write(begin)
        f.write(end)
        f.flush()

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
    def __init__(self, emit_assembly, debug):
        self.emit_assembly = emit_assembly
        self.debug = debug

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
        logging.debug('Read' + str(event))
        assert 'heuristic' in event
        heuristic = int.from_bytes(fc.read(8))
        logging.debug(heuristic)
        fc.readline()
        return heuristic

    def read_action(self, fc):
        event = json.loads(fc.readline())
        logging.debug('Read' + str(event))
        assert 'action' in event
        action = bool(int.from_bytes(fc.read(1)))
        logging.debug(action)
        fc.readline()
        return action

    def get_unroll_decision_results(
            self,
            mod,
            process_and_args,
            working_dir: str):

        self.channel_base = os.path.join(working_dir, 'channel')
        self.to_compiler = self.channel_base + ".in"
        self.from_compiler = self.channel_base + ".out"

        self.process_and_args = process_and_args
        args_to_add = [
            f'--mlgo-loop-unroll-interactive-channel-base={self.channel_base}',
            '--mlgo-loop-unroll-advisor-mode=development',
        ]
        if self.debug:
            args_to_add.append('-debug-only=loop-unroll-development-advisor')
        if self.emit_assembly:
            args_to_add.append('-S')

        process_name = os.path.split(self.process_and_args[0])[1]
        if 'clang' in process_name:
            args_to_add = sum([[x[0], x[1]] for x in zip(['-mllvm'] * len(args_to_add), args_to_add)], [])
            raise Exception("Clang not supported")
        elif 'opt' in process_name:
            pass
        else:
            raise Exception("Unknown compiler")

        self.process_and_args = self.process_and_args + args_to_add

        try:
            logging.debug(f"Opening pipes {self.to_compiler} and {self.from_compiler}")
            os.mkfifo(self.to_compiler, 0o666)
            os.mkfifo(self.from_compiler, 0o666)

            self.num_decisions, _ = self.compile_once(
                mod,
                self.on_features_collect,
                self.on_heuristic_print,
                self.on_action_print,
                lambda index: None,
                lambda index, tensor, heuristic: make_response_for_factor(heuristic)
               )

            if self.num_decisions is None:
                return

            logging.debug(f'Found {self.num_decisions} decisions to make')
            logging.debug(f'Collected features: {self.features}')

            for decision in range(self.num_decisions):
                logging.debug(f'Exploring decision: {decision}')
                decision_results = []
                # From factor = 1 (i.e. no unroll) to MAX_UNROLL_FACTOR inclusive
                for factor in range(1, MAX_UNROLL_FACTOR + 1):
                    logging.debug(f'Exploring factor: {factor}')
                    _, out_module = self.compile_once(
                        mod,
                        lambda index, features: (),
                        lambda index, heuristic: (),
                        lambda index, action: self.on_action_save(index, action) if index == decision else None,
                        lambda index: ("__mlgo_unrolled_loop_begin", "__mlgo_unrolled_loop_end") if index == decision else None,
                        lambda index, tensor, heuristic: make_response_for_factor(factor) if index == decision else make_response_for_factor(heuristic)
                    )
                    if out_module is None:
                        break
                    decision_results.append(
                        UnrollDecisionResult(factor, self.cur_action, out_module))
                else:
                    # If we did not break the above loop
                    ud = UnrollDecision(self.features[decision], decision_results)
                    logging.debug(pprint.pformat(ud))
                    logging.debug('Got result:')
                    yield ud
                    continue
                break

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
            get_instrument_response,
            get_response):

        cur_decision = 0

        logging.debug(f"Launching compiler {' '.join(self.process_and_args)}")
        compiler_proc = subprocess.Popen(
            self.process_and_args, stderr=subprocess.PIPE,
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
        with io.BufferedWriter(io.FileIO(self.to_compiler, "w+b")) as tc:
            with io.BufferedReader(io.FileIO(self.from_compiler, "r+b")) as fc:

                fl = fcntl.fcntl(fc.fileno(), fcntl.F_GETFL)

                # We need to set the reading pipe to nonblocking for the purpose
                # of peek'ing and checking if it is readable without blocking
                # and watch for the process diyng as well. We rever to blocking
                # mode for the actual communication.

                def set_nonblocking():
                    fcntl.fcntl(fc.fileno(), fcntl.F_SETFL, fl | os.O_NONBLOCK)
                def set_blocking():
                    fcntl.fcntl(fc.fileno(), fcntl.F_SETFL, fl & ~os.O_NONBLOCK)

                def input_available():
                    if len(fc.peek(1)) > 0:
                        return "yes"
                    if compiler_proc.poll() is not None:
                        return "dead"
                    return "no"


                set_nonblocking()
                while True:
                    ia = input_available()
                    if ia == "dead":
                        return None, None
                    if ia == "yes":
                        break

                set_blocking()

                header = log_reader._read_header(fc)
                tensor_specs = header.features
                advice_spec = header.advice
                context = None

                set_nonblocking()
                while True:
                    ia = input_available()
                    if ia == "dead":
                        break
                    if ia == "no":
                        continue

                    set_blocking()

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

                    send_instrument_response(tc, get_instrument_response(cur_decision))

                    cur_decision += 1
                    set_nonblocking()

                set_blocking()

        outs, errs = compiler_proc.communicate()
        logging.debug("Errs")
        logging.debug(errs.decode("utf-8"))
        logging.debug(f"Outs size {len(outs)}")
        status = compiler_proc.wait()
        logging.debug(f"Status {status}")

        if self.emit_assembly:
            outs = outs.decode("utf-8")
            logging.debug("Output module:")
            logging.debug(outs)


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
    # TODO need to hook up input-gen replay here
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

def get_ud_samples(uds: Iterable[UnrollDecision]):
    for ud in uds:
        sample = get_ud_sample(ud)
        if sample is not None:
            yield sample
        else:
            logging.debug(f'Obtained invalid sample')

# The below functions are mainly for testing purposes
def main(args):

    if args.debug:
        logging.set_verbosity(logging.DEBUG)

    # TODO we should specify the opt program or command line outside of this
    process_and_args = [
        'opt', '-O3', '--input-gen-mode=replay_generated',
    ]

    with open(args.module, 'rb') as f, \
         tempfile.TemporaryDirectory(dir=args.temp_dir) as tmpdir:
        mod = f.read()
        decision_results = UnrollCompilerHost(bool(args.emit_assembly), bool(args.debug)).get_unroll_decision_results(mod, process_and_args, tmpdir)

        for uds in get_ud_samples(decision_results):
            logging.debug(f'Obtained sample {uds}')

def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='Compiler host'
    )
    parser.add_argument('--module', required=True)
    parser.add_argument('--temp-dir', default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('-S', dest='emit_assembly', action='store_true', default=False)
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    parse_args_and_run()
