#!/usr/bin/env python3

# Usage: ./particle_count.py path/to/run

import matplotlib.pyplot as plt
import numpy as np
# from pylab import *
import matplotlib
import sys
from glob import glob
from io import StringIO
import re
from tqdm import tqdm
from os.path import join as pjoin
from functools import reduce
import argparse
from aux import *

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str,
                    help='Folder to simulation runs')
parser.add_argument('-f', type=int, nargs='?', default=0,
                    help='From time step')
parser.add_argument('-t', type=int, nargs='?', default=-1,
                    help='To time step')
parser.add_argument('-s', type=int, nargs='?', default=1,
                    help='Time step step')
args = parser.parse_args()

vars = parse_xoopic_input(pjoin(args.folder, 'input.inp'))
dt = vars['timeStep']

# folder = sys.argv[1]

files = glob(pjoin(args.folder, 'xv', '*'))
pat = re.compile('\d+')
files.sort(key=lambda x: int(pat.findall(x)[-1]))

files = files[args.f:args.t:args.s]

n = np.array([int(pat.findall(a)[-1]) for a in files])
n += 1 # XOOPIC diagnostics is actually off by one
t = n*dt
Nt = len(t)

labels = ['Electron', 'Argon']
nSpecies = len(labels)
nFiles = len(files)

count = np.zeros((nSpecies, nFiles), dtype=int)

for i, file in enumerate(tqdm(files)):

    # Should be reasonably fast
    with open(file) as f:
        f.readline() # Skip first line
        data = np.array([line.split()[0] for line in f], int)
        count[:,i] = np.bincount(data)

t_mid = 0.5*(t[1:]+t[:-1])

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(t, count[0])
ax1.plot(t, count[1], '--')
ax1.set_ylabel('Particle count')
ax2.plot(t_mid, count[0,1:]-count[0,:-1])
ax2.plot(t_mid, count[1,1:]-count[1,:-1], '--')
ax2.set_ylabel('Particle count change per time-step')
ax2.set_xlabel('Time [s]')

plt.show()
