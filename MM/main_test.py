#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: imoufid
"""

# %% Modules
import numpy as np
from timeit import time

# OD modules
from init_param import init
from error import error_a as err_a
from error import error_b as err_b

# %% Parameters
#Model parameters
M, N, L, Mp, Np, Lp, ainf, gam = init(0)

# Weight and pole number
Ns = 3                 # nombre de paramètres
Nf = 1000              # nombre de point dans la bande de fréquence
fmin, fmax = [1e0,1e4] # Bande de fréquences ou evaluer |f_th - f_num|

# %% Initialisation
# weights and poles
rksk = [ np.logspace(np.log10(fmin),np.log10(fmax),Ns) * 0.1,
        -np.logspace(np.log10(fmin),np.log10(fmax),Ns)]

# range of frequencies
freq = np.logspace(np.log10(fmin),np.log10(fmin),Nf)
jom = 2 * np.pi * 1j * freq

# Exact solutions
tha = np.zeros(Nf,dtype=complex)
thb = np.zeros(Nf,dtype=complex)
for i in range(Nf):
    tha[i] = ainf * (1 + M / jom[i] + N * (np.sqrt(1 + jom[i] / L) - 1) / jom[i])
    thb[i] = gam - (gam - 1) / (1 + Mp / jom[i] + Np * (np.sqrt(1 + jom[i]/Lp) - 1) / jom[i])

# %% Optimization

print(r"Computation for alpha...")
time_0 = time.time()
err = err_a(rksk, [ainf, M, N, L], tha, freq, 'mse')
time_elapsed = time.time() - time_0
print("error for alpha: ",err)
print("DONE  (%.4f seconds)" % time_elapsed, '\n')

print(r"Computation for beta...")
time_0 = time.time()
err = err_b(rksk, [gam, Mp, Np, Lp], thb, freq, 'mse')
time_elapsed = time.time() - time_0
print("error for beta:  ",err)
print("DONE  (%.4f seconds)" % time_elapsed, '\n')
