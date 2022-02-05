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
from param_approx_a import param_approx_a as param_approx
#from param_approx_b import param_approx_b as param_approx

# %% Parameters
#Model parameters
M, N, L, Mp, Np, Lp, ainf, gam = init(0)

# Weight and pole number
Ns = 3                 # nombre de paramètres
Nf = 1000              # nombre de point dans la bande de fréquence
fmin, fmax = [1e0,1e4] # Bande de fréquences ou evaluer |f_th - f_num|

# %% Initialisation
# error variable
err  = 0

# range of frequencies
freq = np.logspace(np.log10(fmin),np.log10(fmin),Nf)

# weights and poles
rksk = [ np.logspace(np.log10(fmin),np.log10(fmax),Ns) * 0.1,
        -np.logspace(np.log10(fmin),np.log10(fmax),Ns)]

# %% Optimization

print("Computing...")
time_0 = time.time()
err = param_approx(rksk, [ainf,M,N,L], freq, 'mse')
time_elapsed = time.time() - time_0
print("DONE  (%.4f seconds)" % time_elapsed, '\n')

print("error : e =",err)
