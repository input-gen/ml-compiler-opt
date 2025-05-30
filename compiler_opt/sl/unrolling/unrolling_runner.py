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
import logging
import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, BinaryIO, Union, List, Optional, Iterable
from .unroll_model import ADVICE_TENSOR_LEN, UNROLL_FACTOR_OFFSET, MAX_UNROLL_FACTOR

from statistics import geometric_mean as gmean

import gin
import tensorflow as tf

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import log_reader

from input_gen.utils import InputGenReplay, Input

logger = logging.getLogger(__name__)


def send_instrument_response(f: BinaryIO, response: Optional[Tuple[str, str]]):
    if response is None:
        f.write(bytes([0]))
        f.flush()
    else:
        f.write(bytes([1]))
        begin = response[0].encode("ascii") + bytes([0])
        end = response[1].encode("ascii") + bytes([0])
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
        logger.fatal(f"{spec.dtype} not supported")
        assert False

    if isinstance(value, list):
        to_send = (ctype_func * len(value))(*[convert_el_func(el) for el in value])
    else:
        to_send = ctype_func(convert_el_func(value))

    assert f.write(bytes(to_send)) == ctypes.sizeof(ctype_func) * math.prod(spec.shape)
    f.flush()


def make_response_for_factor(factor: int):
    if factor == -1:
        # Special case - we use this to encode "heuristic did not make
        # decision", i.e. std::nullopt
        return [-1.0 for _ in range(ADVICE_TENSOR_LEN)]
    if factor > MAX_UNROLL_FACTOR:
        # Special case - we use this to encode factors larger that what our
        # model supports
        l = [-1.0 for _ in range(ADVICE_TENSOR_LEN)]
        l[-1] = float(factor)
        return l
    l = [0.5 for _ in range(ADVICE_TENSOR_LEN)]
    if factor == 1 or factor == 0:
        return l
    assert factor <= MAX_UNROLL_FACTOR
    assert factor >= UNROLL_FACTOR_OFFSET
    assert factor - UNROLL_FACTOR_OFFSET < ADVICE_TENSOR_LEN
    l[factor - UNROLL_FACTOR_OFFSET] = 2.0
    return l


@dataclasses.dataclass(frozen=True)
class UnrollFactorResult:
    factor: int
    action: int
    module: bytes


@dataclasses.dataclass(frozen=True)
class UnrollDecision:
    features: List
    results: List[UnrollFactorResult]
    default: UnrollFactorResult


@dataclasses.dataclass(frozen=True)
class CompilationResult:
    module: bytes
    features_spec: List
    advice_spec: List
    num_decisions: int


class UnrollCompilerHost:
    def __init__(self, emit_assembly, debug):
        self.emit_assembly = emit_assembly
        self.debug = debug

        self.cur_action = None
        self.cur_heuristic = None

        self.num_decisions = None
        self.decisions = None

        self.features = []

        self.tensor_mode = "numpy"
        # self.tensor_mode = 'TensorValue'

        self.channel_base = None
        self.to_compiler = None
        self.from_compiler = None

        self.features_spec = None
        self.advice_spec = None

    def on_features_collect(self, index, tensor_values):
        if self.tensor_mode == "numpy":
            tensor_values = [tv.to_numpy() for tv in tensor_values]
        self.features.append(tensor_values)

    def on_heuristic_print(self, index, heuristic):
        logger.debug(f"Got heuristic {heuristic}")

    def on_action_print(self, index, action):
        logger.debug(f"Got action {action}")

    def on_action_save(self, index, action):
        logger.debug(f"Saving action {action}")
        self.cur_action = action

    def on_heuristic_save(self, index, heuristic):
        logger.debug(f"Saving heuristic {heuristic}")
        self.cur_heuristic = heuristic

    def get_replaced_response(self, heuristic, index, factor):
        return make_response_for_factor(heuristic)

    def read_heuristic(self, fc):
        event = json.loads(fc.readline())
        logger.debug("Read" + str(event))
        assert "heuristic" in event
        heuristic = int.from_bytes(fc.read(8), byteorder=sys.byteorder, signed=True)
        logger.debug(heuristic)
        fc.readline()
        return heuristic

    def read_action(self, fc):
        event = json.loads(fc.readline())
        logger.debug("Read" + str(event))
        assert "action" in event
        action = int.from_bytes(fc.read(1), byteorder=sys.byteorder, signed=False)
        logger.debug(action)
        fc.readline()
        return action

    def get_advice_spec(self):
        return self.advice_spec

    def get_features_spec(self):
        return self.features_spec

    def get_unroll_decision_results(self, mod, process_and_args, working_dir: str):
        self.channel_base = os.path.join(working_dir, "channel")
        self.to_compiler = self.channel_base + ".in"
        self.from_compiler = self.channel_base + ".out"

        self.process_and_args = process_and_args
        args_to_add = [
            f"--mlgo-loop-unroll-interactive-channel-base={self.channel_base}",
            "--mlgo-loop-unroll-advisor-mode=development",
        ]
        if self.debug:
            args_to_add.append("-debug-only=loop-unroll-development-advisor")
        if self.emit_assembly:
            args_to_add.append("-S")

        process_name = os.path.split(self.process_and_args[0])[1]
        if "clang" in process_name:
            args_to_add = sum(
                [[x[0], x[1]] for x in zip(["-mllvm"] * len(args_to_add), args_to_add)], []
            )
            raise Exception("Clang not supported")
        elif "opt" in process_name:
            pass
        else:
            raise Exception("Unknown compiler")

        self.process_and_args = self.process_and_args + args_to_add

        cr = self.compile_once(
            mod,
            self.on_features_collect,
            self.on_heuristic_print,
            self.on_action_print,
            lambda index: None,
            lambda index, tensor, heuristic: make_response_for_factor(heuristic),
        )
        if cr is None:
            return
        self.num_decisions = cr.num_decisions
        if self.num_decisions == 0:
            return
        self.features_spec = cr.features_spec
        self.advice_spec = cr.advice_spec

        logger.debug(f"Found {self.num_decisions} decisions to make")
        logger.debug(f"Collected features: {self.features}")

        for decision in range(self.num_decisions):
            logger.debug(f"Exploring decision: {decision}")
            heuristic = None
            decision_results = []
            cur_status = "success"
            # From factor = 1 (i.e. no unroll) to MAX_UNROLL_FACTOR inclusive
            for factor in range(1, MAX_UNROLL_FACTOR + 1):
                logger.debug(f"Exploring factor: {factor}")
                self.cur_action = None
                self.cur_heuristic = None
                cr = self.compile_once(
                    mod,
                    on_features=lambda index, features: (),
                    on_heuristic=(
                        lambda index, heuristic: self.on_heuristic_save(index, heuristic)
                        if index == decision
                        else None
                    ),
                    on_action=(
                        lambda index, action: self.on_action_save(index, action)
                        if index == decision
                        else None
                    ),
                    get_instrument_response=(
                        lambda index: (
                            "__mlgo_unrolled_loop_begin",
                            "__mlgo_unrolled_loop_end",
                        )
                        if index == decision
                        else None
                    ),
                    get_response=(
                        lambda index, tensor, heuristic: make_response_for_factor(factor)
                        if index == decision
                        else make_response_for_factor(heuristic)
                    ),
                )
                if cr is None:
                    cur_status = "compilation_fail"
                    break
                out_module = cr.module

                if factor == 2 and self.cur_action == 0:
                    cur_status = "no_action"
                    break

                assert self.cur_action is not None
                assert self.cur_heuristic is not None
                if heuristic is None:
                    heuristic = self.cur_heuristic
                else:
                    assert heuristic == self.cur_heuristic
                decision_results.append(UnrollFactorResult(factor, self.cur_action, out_module))

            if cur_status == "success":
                # If we did not break the above loop

                # Get the module with instrumented default heuristic decision
                self.cur_action = None
                self.cur_heuristic = None
                cr = self.compile_once(
                    mod,
                    on_features=lambda index, features: (),
                    on_heuristic=(
                        lambda index, heuristic: self.on_heuristic_save(index, heuristic)
                        if index == decision
                        else None
                    ),
                    on_action=(
                        lambda index, action: self.on_action_save(index, action)
                        if index == decision
                        else None
                    ),
                    get_instrument_response=(
                        lambda index: (
                            "__mlgo_unrolled_loop_begin",
                            "__mlgo_unrolled_loop_end",
                        )
                        if index == decision
                        else None
                    ),
                    get_response=lambda index, tensor, heuristic: make_response_for_factor(heuristic),
                )
                assert self.cur_action is not None
                assert self.cur_heuristic is not None
                assert heuristic == self.cur_heuristic

                ud = UnrollDecision(
                    self.features[decision],
                    decision_results,
                    UnrollFactorResult(heuristic, self.cur_action, cr.module),
                )
                logger.debug(pprint.pformat(ud))
                logger.debug("Got result:")
                yield ud
                continue
            elif cur_status == "compilation_fail":
                # There is something seriously wrong and we should not be
                # proceeding
                logger.debug("Compilation failed!")
                break
            elif cur_status == "no_action":
                logger.debug("Got no action")
                continue
            else:
                assert False

    def compile_once(
        self, mod, on_features, on_heuristic, on_action, get_instrument_response, get_response
    ):
        cur_decision = 0

        logger.debug(f"Opening pipes {self.to_compiler} and {self.from_compiler}")

        try:
            os.unlink(self.to_compiler)
        except FileNotFoundError as e:
            pass
        try:
            os.unlink(self.from_compiler)
        except FileNotFoundError as e:
            pass

        os.mkfifo(self.to_compiler, 0o666)
        os.mkfifo(self.from_compiler, 0o666)

        compiler_proc = None
        try:
            logger.debug(f"Launching compiler {' '.join(self.process_and_args)}")
            compiler_proc = subprocess.Popen(
                self.process_and_args,
                stderr=subprocess.DEVNULL if not self.debug else subprocess.PIPE,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
            )
            logger.debug(f"Sending module")
            compiler_proc.stdin.write(mod)

            # FIXME is this the proper way to close the pipe? if we don't set it to
            # None then the communicate call will try to close it again and raise an
            # error
            compiler_proc.stdin.close()
            compiler_proc.stdin = None

            def set_nonblocking(pipe):
                os.set_blocking(pipe.fileno(), False)

            def set_blocking(pipe):
                os.set_blocking(pipe.fileno(), True)

            output_module = b""
            tensor_specs = None
            advice_spec = None

            set_nonblocking(compiler_proc.stdout)

            logger.debug(f"Starting communication")

            with io.BufferedWriter(io.FileIO(self.to_compiler, "w+b")) as tc:
                with io.BufferedReader(io.FileIO(self.from_compiler, "r+b")) as fc:
                    # We need to set the reading pipe to nonblocking for the purpose
                    # of peek'ing and checking if it is readable without blocking
                    # and watch for the process diyng as well. We rever to blocking
                    # mode for the actual communication.

                    def input_available():
                        nonlocal output_module

                        output = compiler_proc.stdout.read()
                        if output is not None:
                            output_module += output
                        if len(fc.peek(1)) > 0:
                            return "yes"
                        if compiler_proc.poll() is not None:
                            return "dead"
                        return "no"

                    set_nonblocking(fc)
                    while True:
                        ia = input_available()
                        if ia == "dead":
                            return None
                        elif ia == "yes":
                            break
                        elif ia == "no":
                            continue
                        else:
                            assert False

                    set_blocking(fc)

                    header = log_reader._read_header(fc)
                    tensor_specs = header.features
                    advice_spec = header.advice
                    context = None

                    set_nonblocking(fc)
                    while True:
                        ia = input_available()
                        if ia == "dead":
                            break
                        elif ia == "yes":
                            ...
                        elif ia == "no":
                            continue
                        else:
                            assert False

                        set_blocking(fc)

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
                            logger.debug(f"context: {last_context}")
                        context = last_context
                        logger.debug(f"observation: {observation_id}")
                        tensor_values = []
                        for fv in features:
                            logger.debug(fv.to_numpy())
                            tensor_values.append(fv)

                        on_features(cur_decision, tensor_values)
                        heuristic = self.read_heuristic(fc)
                        on_heuristic(cur_decision, heuristic)
                        send(tc, get_response(cur_decision, tensor_values, heuristic), advice_spec)
                        on_action(cur_decision, self.read_action(fc))
                        send_instrument_response(tc, get_instrument_response(cur_decision))

                        cur_decision += 1
                        set_nonblocking(fc)

                    set_blocking(fc)

            set_blocking(compiler_proc.stdout)

            outs, errs = compiler_proc.communicate()
            outs = output_module + outs
            if self.debug:
                logger.debug("Errs")
                logger.debug(errs.decode("utf-8"))
            logger.debug(f"Outs size {len(outs)}")
            status = compiler_proc.wait()
            logger.debug(f"Status {status}")
            if status != 0:
                return None

            if self.emit_assembly:
                outs = outs.decode("utf-8")
                logger.debug("Output module:")
                logger.debug(outs)

            return CompilationResult(
                module=outs,
                features_spec=tensor_specs,
                advice_spec=advice_spec,
                num_decisions=cur_decision,
            )
        finally:
            if compiler_proc is not None:
                compiler_proc.kill()


def DUMP_MODULE(module, ident=""):
    print(f"DUMPING_MODULE {ident}")
    dis_command_vector = ["llvm-dis", "-"]
    with subprocess.Popen(
        dis_command_vector, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.PIPE
    ) as dis_process:
        output = dis_process.communicate(input=module)[0].decode("utf-8")
    print(output)


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    process_and_args = [
        "opt",
        "-O3",
    ]

    with open(args.module, "rb") as f:
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = f.read()
            uch = UnrollCompilerHost(args.emit_assembly, args.debug)
            decision_results = uch.get_unroll_decision_results(mod, process_and_args, tmpdir)

            for uds in generate_samples(decision_results, [], dict()):
                logger.info(f"Obtained sample {uds}")


def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Compiler host")
    parser.add_argument("--module", required=True)
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("-S", dest="emit_assembly", action="store_true", default=False)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    parse_args_and_run()
