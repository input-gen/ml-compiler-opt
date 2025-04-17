#!/usr/bin/env python3

import unittest
import tempfile
import sqlite3
import ray
import logging
import pickle

from com_pile_utils.dataset_writer import DatasetWriter, ProcessResult

logger = logging.getLogger(__name__)


def dumps(s):
    return {(i, pickle.dumps(d)) for i, d in s}


class DatasetWriterTest(unittest.TestCase):
    def test_dsw(self):
        with tempfile.NamedTemporaryFile() as fp:
            dbname = fp.name

            data = list(enumerate([None, ["d"], None, ["a", "b"], []]))
            ds = iter(data)
            dw = DatasetWriter(dbname)
            self.assertEqual(dw.already_processed, set())
            dw.process(ds, process_module_wrapper, None)

            con = sqlite3.connect(dbname)
            cur = con.cursor()
            processed = cur.execute("SELECT * FROM processed").fetchall()
            self.assertEqual(set(processed), {(2, 0), (0, 0), (4, 1), (3, 1), (1, 1)})
            d = cur.execute("SELECT * FROM data").fetchall()
            self.assertEqual(set(d), dumps({(3, "a"), (3, "b"), (1, "d")}))
            con.close()

            data = list(enumerate([None, ["dssssssssssss"], ["c"], ["aaaaaaaaaa"], []]))
            ds = iter(data)
            dw = DatasetWriter(dbname)
            self.assertEqual(dw.already_processed, set(range(5)))
            dw.process(ds, process_module_wrapper, None)

            con = sqlite3.connect(dbname)
            cur = con.cursor()
            processed = cur.execute("SELECT * FROM processed").fetchall()
            self.assertEqual(set(processed), {(2, 0), (0, 0), (4, 1), (3, 1), (1, 1)})
            d = cur.execute("SELECT * FROM data").fetchall()
            self.assertEqual(set(d), dumps({(3, "a"), (3, "b"), (1, "d")}))
            con.close()

            data = [(10, ["e"])]
            ds = iter(data)
            dw = DatasetWriter(dbname)
            self.assertEqual(dw.already_processed, set(range(5)))
            dw.process(ds, process_module_wrapper, None)

            con = sqlite3.connect(dbname)
            cur = con.cursor()
            processed = cur.execute("SELECT * FROM processed").fetchall()
            self.assertEqual(set(processed), {(2, 0), (0, 0), (4, 1), (3, 1), (1, 1), (10, 1)})
            d = cur.execute("SELECT * FROM data").fetchall()
            self.assertEqual(set(d), dumps({(3, "a"), (3, "b"), (1, "d"), (10, "e")}))
            con.close()

            dw = DatasetWriter(dbname)
            self.assertEqual(dw.already_processed, set(range(5)).union({10}))


@ray.remote
def process_module_wrapper(args, idx, data):
    return ProcessResult(idx, data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
