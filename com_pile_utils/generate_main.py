import logging

from .dataset_reader import DatasetReader
from .dataset_writer import DatasetWriter, ID_FIELD


def main(args, process_module_wrapper):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.one is None:
        with DatasetWriter(args.output_dataset) as dw:
            with DatasetReader(args.dataset, dw.get_already_processed()) as dr:
                dw.process(dr.get_iter(), process_module_wrapper, args)
    else:
        with DatasetReader(args.dataset) as dr:
            it = dr.get_one_iter(args.one)
            process_module(args, args.one, next(it)[1])
