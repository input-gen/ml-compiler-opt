import unittest
import sys
import datasets
import ray
import logging
import generate_com_pile_loop_inputs
import generate_com_pile_loop
from argparse import Namespace

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

from input_gen.utils import (
    InputGenReplay,
    InputGenGenerate,
    Input,
    InputGenError,
    InputGenInstrumentationError,
)

TEST_MODULE = b"""
define dso_local void @_Z8vec_initPdi(ptr noundef captures(none) %a, i32 noundef %n) local_unnamed_addr #0 {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds nuw double, ptr %a, i64 %indvars.iv
  store double 3.010400e+04, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
attributes #0 = { inputgen_entry }
"""


# class InputGenTest(tf.test.TestCase):
class GenCompileLoopInputsTest(unittest.TestCase):
    def test_input_gen(self):
        args = Namespace(
            mclang=[],
            mllvm=[],
            save_temps=False,
            temp_dir=None,
        )
        data = dict()
        data["content"] = TEST_MODULE
        data["language"] = "c"
        loops = generate_com_pile_loop.process_module_wrapper_local(args, 10, data)
        self.assertIsNotNone(loops)
        logger.debug("loops")
        logger.debug(loops)
        self.assertEqual(loops.i, 10)
        ds = datasets.Dataset.from_pandas(loops.df)
        logger.debug(ds)
        loop = ds[0]
        logger.debug("loop")
        logger.debug(loop)

        args = Namespace(
            mclang=[],
            mllvm=[],
            save_temps=False,
            temp_dir=None,
            debug_instrumentation=False,
        )
        loop_inputs = generate_com_pile_loop_inputs.process_module(args, 13, loop)
        self.assertIsNotNone(loop_inputs)
        self.assertEqual(loop_inputs.i, 13)
        logger.debug(loop_inputs.df)


if __name__ == "__main__":
    # tf.test.main()
    unittest.main()
