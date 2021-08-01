import os
import mxnet as mx
import numpy as np
import argparse
from tools import boolean_string

class DataLoader:
    """
    data loader for liver_hepg2 r18 proein abundance experiment
    """
    def _get_argparse():
        parser = argparse.ArgumentParser()
        parser.add_argument(
                            '--rna', type=str,
                            help='path to transcriptom out'
                            )
        parser.add_argument(
                            '--prot1D', type=str,
                            help='path to prot1D out'
                            )
        parser.add_argument(
                            '--prot2D', type=str,
                            help='path to prot2D out'
                            )
        parser.add_argument(
                            '--ionData', type=str,
                            help='path to file with ion data on different compounds'
                            )
        parser.add_argument(
                            '--geneIds', type=str,
                            help='path to gene ids description'
                            )
        return parser.parse_args()