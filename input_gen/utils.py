#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tool for generating inputs for an llvm module"""

import signal
import shutil
import argparse
import tempfile
import re
import os
import json
import subprocess
import dataclasses
import collections
import sys
import stat
import logging
import glob
import time
from typing import Dict, Tuple, BinaryIO, Union, List, Optional, Iterable

logger = logging.getLogger(__name__)

ANY = "(.*)"
UINT_REGEX = "([0-9]+)"
INT_REGEX = "(-?[0-9]+)"
DOT = "\\."
RE_MATCH_TIMER = re.compile(f"InputGenTimer {ANY}: {UINT_REGEX} nanoseconds")
RE_MATCH_TIMER_NAME = re.compile(f"{ANY} {UINT_REGEX}")
RE_MATCH_INPUT_FILENAME = re.compile(
    f"{ANY}{DOT}{UINT_REGEX}{DOT}{UINT_REGEX}{DOT}{INT_REGEX}{DOT}{UINT_REGEX}{DOT}inp"
)


@dataclasses.dataclass(frozen=True)
class Input:
    entry_no: int
    index: int
    status: int
    seed: int
    timers: Dict
    data: bytes
    replay_time: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class ReplayResult:
    outs: bytes
    errs: bytes
    timers: Dict


class InputGenError(Exception):
    pass


class InputGenExecError(InputGenError):
    cmd: List[str]
    outs: str
    errs: str

    def __init__(self, cmd, outs, errs):
        self.cmd = cmd
        self.outs = outs
        self.errs = errs
        super().__init__(
            "cmd:\n{}\nouts:\n{}\nerrs:\n{}\n".format(" ".join(self.cmd), self.outs, self.errs)
        )


class InputGenInstrumentationError(InputGenExecError):
    pass


class InputGenTimeout(InputGenError):
    pass


def log_output(outs, errs):
    logger.debug("Logging output")
    if outs is not None:
        outs = outs.decode("utf-8")
        logger.debug(f"Outs: {outs}")
    if outs is not None:
        errs = errs.decode("utf-8")
        logger.debug(f"Errs: {errs}")


def terminate_proc_sns(proc):
    os.killpg(proc.pid, signal.SIGTERM)


def kill_proc_sns(proc):
    os.killpg(proc.pid, signal.SIGKILL)


def terminate_proc(proc):
    proc.terminate()


def kill_proc(proc):
    proc.kill()


class InputGenUtils:
    def __init__(
        self,
        working_dir=None,
        save_temps=False,
        mclang=None,
        mllvm=None,
        temp_dir=None,
        compile_timeout=None,
    ):
        self.save_temps = save_temps
        self.compile_timeout = compile_timeout

        if mclang is not None:
            self.mclang = mclang
        else:
            self.mclang = []
        if mllvm is not None:
            self.mllvm = mllvm
        else:
            self.mllvm = []

        self.save_temps_counter = 0

        self.working_dir = working_dir
        self.temp_dir_arg = temp_dir

    def prepare_utils(self):
        if self.working_dir is not None:
            self.temp_dir = None
        else:
            # FIXME the delete= kw is not valid in python <=3.11 so we cant use
            # it. on the other hand the ml-compiler-opt infra does not support
            # python >=3.12 so we cannot use it until we get support for >=3.12
            #
            # self.temp_dir = tempfile.TemporaryDirectory(dir=temp_dir, delete=(not save_temps))
            # self.working_dir = self.temp_dir.name

            self.temp_dir = tempfile.mkdtemp(dir=self.temp_dir_arg)
            self.working_dir = self.temp_dir

    def get_compile_timeout(self):
        return self.compile_timeout

    def cleanup(self):
        if self.temp_dir is not None and not self.save_temps:
            shutil.rmtree(self.temp_dir)

    def save_temp(self, content, name="temp", binary=True):
        if not self.save_temps:
            return

        self.save_temps_counter += 1
        if binary:
            mode = "wb"
        else:
            mode = "w"
        fn = os.path.join(self.working_dir, str(self.save_temps_counter) + "_intermediate_" + name)
        logger.info(f"Saving temp {fn}")
        with open(fn, mode) as f:
            f.write(content)

    def get_output(
        self,
        cmd,
        stdin=None,
        allow_fail=False,
        env=None,
        timeout=None,
        ExecFailTy=InputGenExecError,
    ):
        logger.debug(f"Running cmd: {' '.join(cmd)}")

        # sns = False if cmd[0] != "clang++" else True
        # Only clang++ needs it but just in case let's use a process group for everything
        sns = True

        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=(subprocess.PIPE if stdin is not None else None),
            env=env,
            start_new_session=sns,
        ) as proc:
            try:
                outs, errs = proc.communicate(input=stdin, timeout=timeout)
                status = proc.wait()
            except subprocess.TimeoutExpired as e:
                if sns:
                    kill_fn = kill_proc_sns
                    terminate_fn = terminate_proc_sns
                else:
                    kill_fn = kill_proc
                    terminate_fn = terminate_proc

                logger.debug("Process timed out! Terminating...")
                terminate_fn(proc)
                try:
                    proc.communicate(timeout=1)
                except subprocess.TimeoutExpired as e:
                    logger.debug("Termination timed out! Killing...")
                    kill_fn(proc)
                    proc.communicate()

                    logger.debug("Killed.")
                    raise InputGenTimeout(f"Timed out: {cmd}")

                logger.debug("Terminated.")
                raise InputGenTimeout(f"Timed out: {cmd}")

            if status != 0 and not allow_fail:
                logger.debug(f"Exit with status {status}")
                logger.debug(f"cmd: {' '.join(cmd)}")
                logger.debug(f"output:")
                errs_decoded = errs.decode("utf-8")
                logger.debug(errs_decoded)

                logger.debug("Failed.")
                raise ExecFailTy(cmd, outs.decode("utf-8"), errs_decoded)

            logger.debug("Finished.")
            return outs, errs

    def get_entry_args(self):
        cmd = []
        if self.entries == "all":
            cmd.append("--input-gen-entry-all-functions")
        elif self.entries == "marked":
            pass
        else:
            for func_name in self.entries:
                cmd.append("--input-gen-entry-function=" + func_name)
        return cmd

    def get_executable_for_generation(self, mod, path):
        cmd = ["opt", "-O3", "--input-gen-mode=generate"] + self.mllvm + self.get_entry_args()
        instrumented_mod, _ = self.get_output(
            cmd,
            mod,
            ExecFailTy=InputGenInstrumentationError,
            timeout=self.get_compile_timeout(),
        )
        self.save_temp(instrumented_mod, "instrumented_mod_for_generation.bc", binary=True)
        with tempfile.NamedTemporaryFile(dir=self.working_dir, suffix=".bc", delete=False) as f:
            f.write(instrumented_mod)
            f.flush()
            cmd = [
                "clang++",
                f.name,
                "-linputgen.generate",
                "-lpthread",
                "-fuse-ld=lld",
                "-O3",
                "-flto",
                "-o",
                path,
            ] + self.mclang
            outs, errs = self.get_output(
                cmd,
                instrumented_mod,
                ExecFailTy=InputGenInstrumentationError,
                timeout=self.get_compile_timeout(),
            )
        log_output(outs, errs)

    def get_no_opt_replay_module(self, mod):
        cmd = (
            [
                "opt",
                "-passes=input-gen-instrument-entries,input-gen-instrument-memory",
                "--input-gen-mode=replay_generated",
            ]
            + self.mllvm
            + self.get_entry_args()
        )
        mod, _ = self.get_output(
            cmd,
            mod,
            ExecFailTy=InputGenInstrumentationError,
            timeout=self.get_compile_timeout(),
        )
        self.save_temp(mod, "instrumented_no_opt_replay_module.bc", binary=True)
        return mod

    def get_executable_for_replay_no_opt(self, mod, path):
        with tempfile.NamedTemporaryFile(dir=self.working_dir, suffix=".bc", delete=False) as f:
            f.write(mod)
            f.flush()
            cmd = ["clang++", f.name, "-linputgen.replay", "-lpthread", "-o", path] + self.mclang
            exe, _ = self.get_output(
                cmd,
                mod,
                ExecFailTy=InputGenInstrumentationError,
                timeout=self.get_compile_timeout(),
            )


class InputGenReplay(InputGenUtils):
    def __init__(
        self,
        mod,
        working_dir=None,
        save_temps=False,
        mclang=None,
        mllvm=None,
        temp_dir=None,
        compile_timeout=None,
    ):
        self.mod = mod

        self.num_entries = None
        self.preparation_done = False

        self.repl_exec = None

        self.inputs_written = 0

        super().__init__(working_dir, save_temps, mclang, mllvm, temp_dir, compile_timeout)

    def __enter__(self):
        self.prepare_utils()
        self.prepare()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def check_prep_done(self):
        if not self.preparation_done:
            raise InputGenError("Preparation not done")

    def check_entry_no(self, entry_no):
        if entry_no < 0 or entry_no >= self.num_entries:
            raise InputGenError(f"Entry no {entry_no} out of range. {self.num_entries} available.")

    def prepare(self):
        assert not self.preparation_done
        self.save_temp(self.mod, "original_module", binary=True)
        self.repl_exec_path = os.path.join(self.working_dir, "repl")
        self.repl_exec = self.get_executable_for_replay_no_opt(self.mod, self.repl_exec_path)

        cmd = [self.repl_exec_path]
        _, errs = self.get_output(cmd, allow_fail=True, timeout=self.get_compile_timeout())
        errs = errs.decode("utf-8")
        re_match = re.search("  Num available functions: ([0-9]+)", errs)

        if re_match is None:
            raise InputGenError("Could not parse number of available entries\n" + errs)

        self.num_entries = int(re_match.group(1))
        self.preparation_done = True

    def get_num_entries(self):
        self.check_prep_done()
        return self.num_entries

    def replay_input(self, inpt, entry_no=0, num=1, timeout=None):
        self.check_prep_done()
        self.check_entry_no(entry_no)

        # TODO keep track of files we've written to disk? or return an input
        # 'handle' which can then be used to replay?
        fn = os.path.join(self.working_dir, f"{entry_no}.{self.inputs_written}.inp")
        self.inputs_written += 1
        with open(fn, "wb") as f:
            f.write(inpt)

        return self.replay_input_path(fn, entry_no, num, timeout)

    def replay_input_path(self, inpt_path, entry_no=0, num=1, timeout=None):
        self.check_prep_done()
        self.check_entry_no(entry_no)

        # TODO it would be really nice if we can tell the replay binary to
        # replay it `num` times so we don't have to relaunch the process every
        # time.
        cmd = [
            self.repl_exec_path,
            inpt_path,
            str(entry_no),
        ]

        i = 0
        while True:
            if num is not None and i == num:
                break
            outs, errs = self.get_output(cmd, allow_fail=True, timeout=timeout)
            timers = dict()
            for timer_name, timer_time in RE_MATCH_TIMER.findall(errs.decode("utf-8")):
                timers[timer_name] = int(timer_time)
            yield ReplayResult(outs, errs, timers)
            i += 1


class InputGenGenerate(InputGenUtils):
    def __init__(
        self,
        mod,
        working_dir=None,
        save_temps=False,
        mclang=None,
        mllvm=None,
        entries="marked",
        temp_dir=None,
        compile_timeout=None,
    ):
        self.mod = mod
        self.entries = entries

        self.num_entries = None
        self.preparation_done = False

        self.gen_exec = None
        self.repl_mod = None

        super().__init__(working_dir, save_temps, mclang, mllvm, temp_dir, compile_timeout)

    def __enter__(self):
        self.prepare_utils()
        self.prepare()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def prepare(self):
        self.save_temp(self.mod, "original_module", binary=True)
        self.gen_exec_path = os.path.join(self.working_dir, "gen")
        self.gen_exec = self.get_executable_for_generation(self.mod, self.gen_exec_path)
        self.repl_mod = self.get_no_opt_replay_module(self.mod)

        cmd = [self.gen_exec_path, "-1"]
        _, errs = self.get_output(cmd, allow_fail=True, timeout=self.get_compile_timeout())
        re_match = re.search("  Num available functions: ([0-9]+)", errs.decode("utf-8"))

        if re_match is None:
            raise InputGenError("Could not parse number of available entries")

        self.num_entries = int(re_match.group(1))
        self.preparation_done = True

    def check_prep_done(self):
        if not self.preparation_done:
            raise InputGenError("Preparation not done")

    def generate_batched(
        self,
        entry_no=0,
        num_inputs=1,
        num_threads=1,
        first_input=0,
        seed=42,
        int_min=-100,
        int_max=128,
        timeout=None,
    ):
        self.check_prep_done()
        try:
            cmd = [
                self.gen_exec_path,
                str(entry_no),
                str(num_inputs),
                str(num_threads),
                str(first_input),
                str(seed),
            ]
            env = os.environ
            env["INPUTGEN_INT_MIN"] = str(int_min)
            env["INPUTGEN_INT_MAX"] = str(int_max)
            outs, errs = self.get_output(cmd, env=env, timeout=timeout)
            try:
                outs = outs.decode("utf-8")
                errs = errs.decode("utf-8")
            except UnicodeDecodeError as e:
                logger.debug(e)
                return []

            logger.debug(f"Outs: {outs}")
            logger.debug(f"Errs: {errs}")

            input_idxs = set(range(first_input, first_input + num_inputs))
            input_timers = {i: dict() for i in input_idxs}
            for timer_name, timer_time in RE_MATCH_TIMER.findall(errs):
                timer_idx = None
                name_match = RE_MATCH_TIMER_NAME.fullmatch(timer_name)
                if name_match is not None:
                    timer_idx = int(name_match.group(2))
                    timer_name = name_match.group(1)
                    try:
                        input_timers[timer_idx][timer_name] = int(timer_time)
                    except KeyError as e:
                        keyErrorLogger = logging.getLogger(__name__ + ".timer_key_error")
                        keyErrorLogger.setLevel(logging.INFO)
                        keyErrorLogger.error(
                            f"first_input {first_input} num_inputs {num_inputs} key {timer_idx}"
                        )
                        keyErrorLogger.error(f"cmd {cmd}")
                        keyErrorLogger.error("errs")
                        keyErrorLogger.error(errs)
                        return []
                else:
                    for d in input_timers.values():
                        d[timer_name] = int(timer_time)
            logger.debug(input_timers)

            # TODO we can accidentally grab an earlier generated input if
            # save_temps is on or if we fail to delete for any reason - we
            # should prbably wrap this whope thing in a temp directory create
            # delete for each batched input generation

            inputs = []
            for filename in os.listdir(self.working_dir):
                re_match = RE_MATCH_INPUT_FILENAME.fullmatch(filename)
                logger.debug(filename)
                logger.debug(re_match)
                if re_match is None:
                    continue
                logger.debug(re_match.groups())
                inpt_gen_exe = re_match.group(1)
                inpt_entry_no = int(re_match.group(2))
                inpt_input_idx = int(re_match.group(3))
                inpt_exit_code = int(re_match.group(4))
                inpt_seed = int(re_match.group(5))

                if any(
                    [
                        inpt_gen_exe != "gen",
                        inpt_entry_no != entry_no,
                        inpt_seed != seed,
                        inpt_input_idx not in input_idxs,
                    ]
                ):
                    continue

                full_path = os.path.join(self.working_dir, filename)
                with open(full_path, "rb") as f:
                    data = f.read()

                inputs.append(
                    Input(
                        entry_no,
                        inpt_input_idx,
                        inpt_exit_code,
                        seed,
                        input_timers[inpt_input_idx],
                        data,
                    )
                )

            return inputs

        except InputGenError as e:
            raise e
        finally:
            for filename in os.listdir(self.working_dir):
                re_match = RE_MATCH_INPUT_FILENAME.fullmatch(filename)
                if re_match is None:
                    continue
                full_path = os.path.join(self.working_dir, filename)
                if not self.save_temps:
                    logger.debug(f"Removing {full_path}")
                    os.remove(full_path)

    def generate(
        self, entry_no=0, num_inputs=1, first_input=0, seed=42, int_min=-100, int_max=100, timeout=None
    ):
        self.check_prep_done()
        inputs = []
        for input_idx in range(first_input, first_input + num_inputs):
            inpt = self.generate_batched(
                entry_no=entry_no,
                num_inputs=1,
                num_threads=1,
                first_input=first_input,
                seed=seed,
                timeout=timeout,
            )
            inputs += inpt
        return inputs

    def get_num_entries(self):
        self.check_prep_done()
        return self.num_entries

    def get_repl_mod(self):
        self.check_prep_done()
        return self.repl_mod


def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Generating inputs for a module")
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--save-temps", action="store_true", default=False)
    parser.add_argument("-mclang", default=[], action="append")
    parser.add_argument("-mllvm", default=[], action="append")

    parser.add_argument("--module", required=True)
    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument("--entry-function", default=[], action="append")
    mutex.add_argument("--entry-all", default=False, action="store_true")
    mutex.add_argument("--entry-marked", default=False, action="store_true")

    parser.add_argument("-debug", default=False, action="store_true")

    args = parser.parse_args()
    main(args)


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if int(args.entry_marked) + int(args.entry_all) + int(len(args.entry_function) > 0) != 1:
        logger.error(
            "Exactly one of `--entry-function`, `--entry-all`, or `--entry-marked` must be specified"
        )
        return

    mode = "invalid"
    if args.entry_marked:
        mode = "marked"
    elif args.entry_all:
        mode = "all"
    else:
        mode = args.entry_function

    with open(args.module, "rb") as f:
        mod = f.read()

    # Generate inputs
    with InputGenGenerate(
        mod, args.temp_dir, args.save_temps, args.mclang, args.mllvm, mode, args.temp_dir
    ) as igg:
        inputs = igg.generate()

    # Get generated inputs and module for replay
    replay_module = igg.get_repl_mod()

    with InputGenReplay(
        replay_module, args.temp_dir, args.save_temps, args.mclang, args.mllvm, args.temp_dir
    ) as igr:
        for inpt in inputs:
            num = 5
            timeout = 0.2
            for res in igr.replay_input(inpt.data, inpt.entry_no, num, timeout=timeout):
                logger.info(f"Res {res}")


if __name__ == "__main__":
    parse_args_and_run()
