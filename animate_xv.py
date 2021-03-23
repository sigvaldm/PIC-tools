#!/usr/bin/env python3

# Usage: ./animate_xv.py path/to/run

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
import argparse
from tasktimer import TaskTimer # pip install TaskTimer

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

folder = sys.argv[1]

files = glob(pjoin(folder, 'xv', '*'))
pat = re.compile('\d+')
files.sort(key=lambda x: int(pat.findall(x)[-1]))

files = files[args.f:args.t:args.s]

labels = ['Electron', 'Argon']
nSpecies = len(labels)
fig,axs = plt.subplots(nSpecies,1)

timer = TaskTimer()

ymin = np.inf*np.ones(nSpecies)
ymax = -np.inf*np.ones(nSpecies)

moviewriter = matplotlib.animation.FFMpegWriter(fps=10)
with moviewriter.saving(fig, pjoin(folder, 'xv.mp4'), 100):

    for file in timer.iterate(files):

        timer.task('Read file')

        # This method reads files at half the time of np.loadtxt()
        with open(file) as f:
            f.readline() # Skip first line
            data = np.array([line.strip().split() for line in f], float)

        for i in range(nSpecies):

            timer.task('Plot particles')

            ind, = np.where(data[:,0]==i)

            ymin[i] = min(ymin[i], np.percentile(data[ind,3], 1))
            ymax[i] = max(ymax[i], np.percentile(data[ind,3], 99))

            axs[i].cla()
            axs[i].scatter(data[ind,2],data[ind,3],s=1,marker='.',color='b',alpha=0.6)
            axs[i].set_xlabel("$x [m]$")
            axs[i].set_ylabel("$v [m/s]$")
            axs[i].set_xlim([0,1.5])
            axs[i].set_ylim([0.9*ymin[i], 1.1*ymax[i]])
            axs[i].set_title(labels[i])

        timer.task('Save frame')

        plt.tight_layout()
        moviewriter.grab_frame()

print(timer)
