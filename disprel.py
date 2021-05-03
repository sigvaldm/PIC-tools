#!/usr/bin/env python3

# Usage: ./disprel.py path/to/folder

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mp
import numpy as np
import sys
import os.path
from os.path import join as pjoin
from scipy.constants import value as constants
from glob import glob
import re
from tqdm import tqdm
from parsers import *
import argparse


parser = argparse.ArgumentParser(description='Plasma Dispersion Processor')
parser.add_argument('-i','--input', default=None, type=str, help='Path to E-field data')
parser.add_argument('-per','--periodic', action='store_true', help='Add this if the system is periodic in Y')
parser.add_argument('-yLoc','--yLocation', default=1, type=int, help='In bounded (in Y) system Choose Y location, Options: e.g. 1 (Any number between 0-Ny)')
parser.add_argument('-pl','--plot', action='store_true', help='Add this if you want to plot the figure')
parser.add_argument('-n','--norm', default='omega_pe', type=str, help='Normalizing frequency, Options: omega_pi, omega_pe')

args        = parser.parse_args()
folder      = args.input
periodic    = args.periodic
yLoc        = args.yLocation	#8 # Choose y location
plot        = args.plot
norm        = args.norm

# Set processed data directory
folder_base= os.path.basename(os.path.dirname(folder))
savedir     = pjoin(folder, '..',folder_base+'_processed')

files = glob(pjoin(folder, 'Ex_*.dat'))
pat = re.compile('\d+')
files.sort(key=lambda x: int(pat.findall(x)[-1]))

# READ TEMPORAL GRID
vars = parse_xoopic_input(pjoin(folder, '..', 'input.inp'))
# print(vars['Control'][0]['dt'])
dt = vars['Control'][0]['dt']

n = np.array([int(pat.findall(a)[-1]) for a in files])
n += 1 # XOOPIC diagnostics is actually off by one
t = n*dt
Nt = len(t)

if  os.path.exists(savedir) and os.path.exists(pjoin(savedir,'pro_data_file.npz')):
	print('processed data exists. Loading data ...')
	f = np.load(pjoin(savedir,'pro_data_file.npz'))['data']
	print('Shape of loaded data fp: ',f.shape)
	x = np.load(pjoin(savedir,"x.npz"))['x']
	print("Shape of loaded data x",x.shape)

else:
  if  os.path.exists(savedir)== False:
    os.mkdir(savedir)
  print('processed data does not exist. Wait, processing data ...')

  # READ SPATIAL GRID
  data = np.loadtxt(files[0])

  x, y = data[:,2:4].T
  Ny = len(np.where(x==x[0])[0])
  Nx = len(x)//Ny
  x = x.reshape(Nx, Ny)
  y = y.reshape(Nx, Ny)

  # READ FIELD
  f = np.zeros((Nt, Nx, Ny))

  for i, file in enumerate(tqdm(files)):
    data = np.loadtxt(file)
    f[i] = data[:,-1].reshape(Nx, Ny)

    # Remove y
  if periodic == False:
      f = f[:,:,yLoc-4:yLoc+4]
  f = np.average(f, axis=2)
  x = np.average(x, axis=1)
  np.savez_compressed(pjoin(savedir,'pro_data_file.npz'),data=f)
  np.savez_compressed(pjoin(savedir,"x.npz"),x=x)

if plot:
    # FFT axes
  # dt = vars['timeStep']*vars['save_step'] #t[1]-t[0]

  Mt = len(t)
  Mx = vars['Grid'][0]['J'] #len(x)
  Lx = vars['Grid'][0]['x1f'] - vars['Grid'][0]['x1s']
  dx = Lx/Mx #x[1]-x[0]
  # dt = t[1]-t[0]
  # dx = x[1]-x[0]
  # Mt = len(t)
  # Mx = len(x)
  omega = 2*np.pi*np.arange(Mt)/(Mt*dt)
  k     = 2*np.pi*np.arange(Mx)/(Mx*dx)
  print('Length of k: ',len(k))
  print('Max of k: ',np.max(k))
  Omega, K = np.meshgrid(omega, k, indexing='ij')
  print('Shape of Omega: ',Omega.shape)
  F = np.fft.fftn(f, norm='ortho')

  halflen = np.array(F.shape, dtype=int)//2
  Omega = Omega[:halflen[0],:halflen[1]]
  K = K[:halflen[0],:halflen[1]]
  F = F[:halflen[0],:halflen[1]]

  # Analytical ion-acoustic dispresion relation
  ne = vars['Load'][0]['density']
  ni = ne

  eps0 = constants('electric constant')
  kb = constants('Boltzmann constant')
  me = constants('electron mass')
  e = constants('elementary charge')

  mi  = vars['Species'][1]['m'] #40*constants('atomic mass constant')
  nK  = vars['Grid'][0]['J']
  gamma_e = 5./3

  units = vars['Load'][0]['units']
  if units == 'MKS':
      vthE = vars['Load'][0]['temperature']
      tEeV   = 0.5*me*(vthE*vthE)/e
      tEK    = tEeV*11604.525
      vthI = vars['Load'][1]['temperature']
      tIeV   = 0.5*mi*(vthI*vthI)/e
      tIK    = tIeV*11604.525

  Te  = tEK #vars['tEK'] #1.6*11604
  Ti  = tIK #vars['tIK'] #0.1*11604
  vb  = vars['Load'][1]['v1drift']

  wpi = np.sqrt(e**2*ni/(eps0*mi))
  wpe = np.sqrt(e**2*ne/(eps0*me))
  dl	= np.sqrt(eps0*kb*Te/(ni*e*e))
  dli	= np.sqrt(eps0*kb*Ti/(ni*e*e))
  print("wpi={}".format(wpi))
  print("dl={}".format(dl))
  cia = np.sqrt(gamma_e*kb*Te/mi)
  ka = np.linspace(0, np.max(K), nK)
  wac = np.sqrt((ka*cia)**2/(1+(ka*cia/wpi)**2))
  wah = np.sqrt( (wpi**2) * (ka*ka * dli*dli * (Te/Ti))/(1+(ka*ka * dli*dli * (Te/Ti))) )
  wl = np.sqrt( (wpe**2) * (1+me/mi) * (1+(3*ka*ka*dl*dl)/(1+me/mi)) )
  # wea =
  wb = ka*vb


  #omega_pe

  if norm == "omega_pi":
    omega /= wpi
    Omega /= wpi
  else:
    omega /=wpe
    Omega /=wpe

  wl /= wpe
  wac /= wpi
  wah /= wpi
  wb /= wpi

  Z = np.log(np.abs(F))
  #Z = np.abs(F)

  # ==== Figure =============

  ##### FIG SIZE CALC ############
  figsize = np.array([150,150/1.618]) #Figure size in mm
  dpi = 300                         #Print resolution
  ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution

  fig,ax = plt.subplots(figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
  mp.rc('text', usetex=False)
  mp.rc('font', family='sans-serif', size=12, serif='Computer Modern Roman')
  mp.rc('axes', titlesize=12)
  mp.rc('axes', labelsize=12)
  mp.rc('xtick', labelsize=12)
  mp.rc('ytick', labelsize=12)
  mp.rc('legend', fontsize=12)

  oRange = len(K[:,0]) #for full omega len(K[:,0])
  # oRange = int(oRange/50)
  print(K[:oRange,:].shape,Omega[:oRange,:].shape,Z[:oRange,:].shape)
  # print(oRange)
  if norm == "omega_pi":
    oRange = int(oRange/200)
    plt.pcolor(K[:oRange,:], Omega[:oRange,:], Z[:oRange,:],shading='auto',vmin=np.min(Z[:oRange,:]),vmax=np.max(Z[:oRange,:])) #np.min(Z[:oRange,:])
    plt.colorbar()
  else:
    oRange = int(oRange/50)
    plt.pcolor(K[:oRange,:], Omega[:oRange,:], Z[:oRange,:],shading='auto',vmin=np.min(Z[:oRange,:]),vmax=np.max(Z[:oRange,:]))
    #plt.pcolor(K, Omega, Z,shading='auto',vmin=np.min(Z),vmax=np.max(Z))
    #plt.imshow(K, Omega, Z)
    plt.colorbar()

  if norm == "omega_pi":
    plt.plot(ka, wb, '--w', label="Beam Mode")
    # plt.plot(ka, wah, '--r',label="IAW with warm ions")
    # plt.plot(ka, wb, '--w',label="Beam driven waves")
    leg = ax.legend()
    ax.set_xlabel('$k~[1/m]$')
    ax.set_ylabel('$\omega/\omega_{pi}$')
  else:
    plt.plot(ka, wb, '--w', label="langmuir wave")
    # plt.axhline(y=1.0, color='w', linestyle='--',label='$\omega_{pe}$')
    leg = ax.legend()
    ax.set_xlabel('$k~[1/m]$')
    ax.set_ylabel('$\omega/\omega_{pe}$')

  ax.set_ylim([0, 2])
  plt.savefig(pjoin(savedir, norm+'_disprel.png'))
  # plt.show()
