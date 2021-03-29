#!/usr/bin/env python3

# Usage: ./disprel.py path/to/folder

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import sys
import os.path
from os.path import join as pjoin
from scipy.constants import value as constants
from glob import glob
import re
from tqdm import tqdm
from aux import *
import argparse


parser = argparse.ArgumentParser(description='Plasma Dispersion Processor')
parser.add_argument('--i', default=None, type=str, help='Path to E-field data')
parser.add_argument('--periodic', default=True, type=bool, help='Period System in Y')
parser.add_argument('--yLoc', default=4, type=int, help='Choose Y location in bounded system')
parser.add_argument('--sN', default=4, type=int, help='No. of chuncks you want to split the processed file')
parser.add_argument('--plot', default=True, type=bool, help='Plot the figure')

args		= parser.parse_args()
folder		= args.i
periodic	= args.periodic
yLoc		= args.yLoc	#8 # Choose y location
splitNos	= args.sN #No. of chunks
plot		= args.plot

runName 	= os.path.basename(os.path.dirname(folder))
print(runName)
files 		= glob(pjoin(folder, 'Ex_*.dat'))
savedir 	= pjoin(folder, '../',runName+'_processed/') #directory for processed data
print(savedir)

pat = re.compile('\d+')
files.sort(key=lambda x: int(pat.findall(x)[-1]))

#Split flie list into multiple chunks
cSize = int(len(files)/splitNos) #Chunck size
files_part = [files[i:i + cSize] for i in range(0, len(files), cSize)]
# READ TEMPORAL GRID

vars = parse_xoopic_input(pjoin(folder, '..', 'input.inp'))
dt = vars['timeStep']

n = np.array([int(pat.findall(a)[-1]) for a in files])
n += 1 # XOOPIC diagnostics is actually off by one
t = n*dt
Nt = len(t)

if  os.path.exists(savedir) and os.path.exists(savedir+'pro_files_part_0.dat'):
	print('processed data exists. Loading data ...')
	fp = []
	for i in range(splitNos):
	    ftemp = np.loadtxt(savedir+"pro_files_part_%d"%i+".dat", unpack=True)
	    print(savedir+"pro_files_part_%d"%i+".dat")
	    ftemp= np.array(ftemp.T)
	    print("Shape of loaded data f1: ",ftemp.shape)
	    fp.append(ftemp)
	fp = np.array(fp)
	print("Shape of final data fp: ",fp.shape)
	# fp = np.average(fp, axis=0)
	# fp=np.vstack((fp[0,:,:],fp[1,:,:],fp[2,:,:],fp[3,:,:]))
	fp = fp.reshape(-1,len(fp[0,0,:]))
	print("Shape of final data vstack fp: ",fp.shape)
	x = np.loadtxt(savedir+"x.dat", unpack=True)
	print("Shape of loaded data x",x.shape)
else:
	print('processed data does not exist. Wait, processing data ...')
	os.mkdir(savedir)
	fp = []# READ SPATIAL GRID
	for i in range(splitNos):
		data = np.loadtxt(files_part[i][0])
		# print("DATA shape: ",len(files_part[i]))
		x, y = data[:,2:4].T
		Ny = len(np.where(x==x[0])[0])
		Nx = len(x)//Ny
		x = x.reshape(Nx, Ny)
		y = y.reshape(Nx, Ny)

		# READ FIELD

		f = np.zeros((len(files_part[i]), Nx, Ny))

		for j, fi in enumerate(tqdm(files_part[i])):
			data = np.loadtxt(fi)
			f[j] = data[:,-1].reshape(Nx, Ny)
		print('Shape of raw f: ',f.shape)
		# Remove y
		#f = np.average(f, axis=2)
		if periodic == False:
			f = f[:,:,yLoc-4:yLoc+4]
		f = np.average(f, axis=2)
		fp.append(f)
		x = np.average(x, axis=1)
		print('Shape of f: ',f.shape)
		np.savetxt(savedir+"pro_files_part_%d"%i+".dat",f,fmt='%.8f')
		np.savetxt(savedir+"x.dat",x,fmt='%.8f')
	fp = np.array(fp)
	print("Shape of data fp: ",fp.shape)
	# fp=np.vstack((fp[0,:,:],fp[1,:,:],fp[2,:,:],fp[3,:,:]))
	fp = fp.reshape(-1,len(fp[0,0,:]))
	print("Shape of data vstack fp: ",fp.shape)

if plot == True:
    # FFT axes
    dt = t[1]-t[0]
    dx = x[1]-x[0]
    Mt = len(t)
    Mx = len(x)
    omega = 2*np.pi*np.arange(Mt)/(Mt*dt)
    k     = 2*np.pi*np.arange(Mx)/(Mx*dx)
    Omega, K = np.meshgrid(omega, k, indexing='ij')

    F = np.fft.fftn(fp, norm='ortho')

    halflen = np.array(F.shape, dtype=int)//2
    Omega = Omega[:halflen[0],:halflen[1]]
    K = K[:halflen[0],:halflen[1]]
    F = F[:halflen[0],:halflen[1]]


    # Analytical ion-acoustic dispresion relation
    ni = 1e13
    ne = ni
    eps0 = constants('electric constant')
    kb = constants('Boltzmann constant')
    me = constants('electron mass')
    mi = 40*constants('atomic mass constant')
    e = constants('elementary charge')
    gamma_e = 5./3
    Te = 11604
    wpi = np.sqrt(e**2*ni/(eps0*mi))
    wpe = np.sqrt(e**2*ne/(eps0*me))
    print('me= ',me)
    print("wpi={}".format(wpi))
    cia = np.sqrt(gamma_e*kb*Te/mi)
    ka = np.linspace(0, np.max(K), 100)
    wa = np.sqrt((ka*cia)**2/(1+(ka*cia/wpi)**2))

    omega /= wpi
    Omega /= wpi
    wa /= wpi

    Z = np.log(np.abs(F))
    #Z = np.abs(F)

    plt.pcolor(K, Omega, Z,shading='auto')
    #plt.imshow(K, Omega, Z)
    plt.colorbar()

    plt.plot(ka, wa, '--w')

    plt.xlabel('k [1/m]')
    plt.ylabel('$\omega/\omega_{pi}$')
    plt.savefig(pjoin(savedir, 'disprel.png'))
    plt.show()

    # fig = plt.figure()
    # ax = plt.gca(projection='3d')
    # plt.imshow(np.log(np.abs(F[-1024:,:])))
    # surf = ax.plot_surface(T, X, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
