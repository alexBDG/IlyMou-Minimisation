#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: imoufid
"""

import numpy as np
from .OD_rep import OD_rep


def param_approx_a(param,param_th,freq,err='mse',file_od=''):
    """
    Objective function for the dynamic tortuosity alpha

    Input variables:
        . param    = [rk,sk] (MM parameters) [list of two arrays],
        . param_th = [O, M, N, L] (Physical parameters) [list of floats]:
            [ainf, M, N, L] for alpha and [gam, Mp, Np, Lp] for beta
        . freq (frequencies) [array of float]: angular frequency w = 2 pi f
        . error (choice of the error to return) [string]: optional
            - 'max': maximal absolute error [float],
            - 'mse': mean squared error [float].
        . file_od (name of the file containing the mm parameters) [string]:
            optional, rk and sk can be obtained by reading an od-type file

    Output variables:
        . err (float): error between the MM approximation and the exact model
    """
    # %% Parameter definition
    nf = len(freq)
    jom = 2 * np.pi * 1j * freq

    mm = np.zeros(nf,dtype=complex)
    th = np.zeros(nf,dtype=complex)

    ainf, M, N, L = param_th

    # %% Error
    if file_od:
        OD = OD_rep(file_od,0)
        mm = OD.fun(jom)
    else:
        mm += param_th[0]
        mm += param_th[0]*param_th[1] / jom
        for k in range(len(param[0])): mm += param[0][k] / (jom - param[1][k])

    for i in range(nf):
        th[i] = ainf * (1 + M / jom[i] + N * (np.sqrt(1 + jom[i] / L) - 1) / jom[i])

    abs_err = np.abs(th-mm)
    if err=='max':
        error = np.max(abs_err)
    elif err=='mse':
        error = np.sum(abs_err*abs_err)/nf
    else:
        print("*** Wrong choice for the error choice ***")
        error = 0

    return error