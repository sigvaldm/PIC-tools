#!/usr/bin/env python3

# Usage: ./summary.py path/to/run another/path/to/run

import numpy as np
from os.path import join as pjoin
import argparse
from aux import *
from langmuir import *

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str,
                    help='Folder to simulation runs')
args = parser.parse_args()

# vars = parse_xoopic_input(pjoin(args.folder, 'input.inp'))
# dt = vars['timeStep']

vars = parse_xoopic(pjoin(args.folder, 'input.inp'))

# omega_pe = Electron(n=vars['nE'], T=vars['tEK']).omega_p

# print(dt/omega_pe)
