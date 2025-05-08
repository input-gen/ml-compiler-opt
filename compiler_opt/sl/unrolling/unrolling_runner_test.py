import unittest
import sys
import datasets
import ray
import logging
from argparse import Namespace
from . import generate_unroll_results as gur
from . import generate_unroll_training_samples as guts
import com_pile_utils.generate_com_pile_loop_inputs as gcpli
import com_pile_utils.generate_com_pile_loop as gcpl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

TEST_MODULE = b"""
define dso_local void @_Z8vec_initPdi(ptr noundef captures(none) %a, i32 noundef %n) local_unnamed_addr #0 {
entry0:
  %cmp1 = icmp sle i32 %n, 10000
  br i1 %cmp1, label %entry, label %for.cond.cleanup
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
        data["id"] = 15
        loops = gcpl.process_module(args, 10, data)
        self.assertIsNotNone(loops)
        logger.debug("loops")
        logger.debug(loops)
        self.assertEqual(loops.i, 10)
        ds = loops.data
        logger.debug(ds)
        loop = ds[0]
        logger.debug("loop")
        logger.debug(loop)
        loop["id"] = 16

        args = Namespace(
            mclang=[],
            mllvm=[],
            save_temps=True,
            temp_dir="./temps1/",
            # save_temps=False,
            # temp_dir=None,
            debug_instrumentation=False,
        )
        loop_inputs = gcpli.process_module(args, 13, loop)
        logger.debug(loop_inputs)
        self.assertIsNotNone(loop_inputs)
        self.assertEqual(loop_inputs.i, 13)
        logger.debug(loop_inputs.data)
        logger.debug(loop_inputs.data[0])
        logger.debug(loop_inputs.data[0]["inputs"])
        self.assertGreater(len(loop_inputs.data[0]["inputs"]), 0)

        loop = loop_inputs.data[0]
        loop["id"] = 17

        args = Namespace(
            mclang=[],
            mllvm=[],
            save_temps=True,
            temp_dir="./temps2/",
            # save_temps=False,
            # temp_dir=None,
            dump_llvm=False,
            debug=True,
        )
        res = gur.process_module(args, 14, loop)
        print(res)
        loop = res.data[0]
        loop["id"] = 18

        args = Namespace(
            mclang=[],
            mllvm=[],
            save_temps=True,
            temp_dir="./temps3/",
            dump_llvm=False,
            debug=False,
        )
        res = guts.process_module(args, 9, loop)
        print(res)


if __name__ == "__main__":
    unittest.main()
