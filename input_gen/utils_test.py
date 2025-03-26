import tensorflow as tf
import unittest
import sys
import logging

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
define i32 @load_from_ptr(ptr %a) #0 {
  %v = load i32, ptr %a
  ret i32 %v
}
define i32 @load_from_ptr_2(ptr %a) #0 {
  %v = load i32, ptr %a
  ret i32 %v
}
attributes #0 = { inputgen_entry }
"""


# class InputGenTest(tf.test.TestCase):
class InputGenTest(unittest.TestCase):
    def test_input_gen(self):
        igg = InputGenGenerate(
            TEST_MODULE,
            mclang=[],
        )
        inputs = igg.generate(entry_no=0, num_inputs=1, first_input=0, seed=43)

        logger.debug(inputs)
        self.assertTrue(len(inputs) == 1)
        inpt = inputs[0]
        self.assertIsNotNone(inpt)
        self.assertTrue(inpt.entry_no == 0)
        self.assertTrue(inpt.index == 0)
        self.assertTrue(inpt.status == 0)
        self.assertTrue(inpt.seed == 43)
        self.assertTrue("init" in inpt.timers)
        self.assertTrue("genr" in inpt.timers)
        self.assertTrue("save" in inpt.timers)
        self.assertTrue(all({i > 0 for i in inpt.timers.values()}))

        repl_mod = igg.get_repl_mod()

        igr = InputGenReplay(
            repl_mod,
            mclang=[],
        )
        for inpt in inputs:
            res = next(igr.replay_input(inpt.data, entry_no=0, num=1))
            logger.debug(res)
            self.assertIsNotNone(res)


if __name__ == "__main__":
    # tf.test.main()
    unittest.main()
