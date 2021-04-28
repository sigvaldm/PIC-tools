#!/usr/bin/env python3

# Usage: ./summary.py path/to/run another/path/to/run

import numpy as np
from os.path import join as pjoin
import argparse
from parsers import *
from langmuir import *
from pprint import PrettyPrinter

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str,
                    help='Folder to simulation runs')
args = parser.parse_args()

config=parse_xoopic_input(pjoin(args.folder, 'input.inp'))
PrettyPrinter().pprint(config)
