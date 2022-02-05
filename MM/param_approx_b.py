#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: imoufid
"""

import numpy as np
from .OD_rep import OD_rep


def param_approx_b(param,param_th,freq,err='mse',file_od=''):
    """
    Objective function for the dynamic compressibility beta

    Input variables:
        . param    = [rk,sk] (MM parameters) [list of two arrays],
        . param_th = [O, M, N, L] (Physical parameters) [list of floats]:
            [ainf, M, N, L] for alpha and [gam, Mp, Np, Lp] for beta
        . freq (frequencies) [array of float]: angular frequency w = 2 pi f
        . file_od (name of the file containing the mm parameters) [string]:
            optional, rk and sk can be obtained by reading an od-type file
        . error (choice of the error to return) [string]: optional
            - 'max': maximal absolute error [float],
            - 'mse': mean squared error [float].

    Output variables:
        . err (float): error between the MM approximation and the JCAPL model
    """
    # %% Parameter definition
    nf = len(freq)
    jom = 2 * np.pi * 1j * freq

    mm = np.zeros(nf,dtype=complex)
    th = np.zeros(nf,dtype=complex)

    gam, Mp, Np, Lp = param_th

    # %% Error
    if file_od:
        OD = OD_rep(file_od,0)
        mm = OD.fun(jom)
    else:
        mm += 1
        for k in range(len(param[0])): mm += param[0][k] / (jom - param[1][k])

    for i in range(nf):
        th[i] = gam - (gam - 1) / (1 + Mp / jom[i] + Np * (np.sqrt(1 + jom[i]/Lp) - 1) / jom[i])

    abs_err = np.abs(th-mm)
    if err=='max':
        error = np.max(abs_err)
    elif err=='mse':
        error = np.sum(abs_err*abs_err)/nf
    else:
        print("*** Wrong choice for the error choice ***")
        error = 0

    return error


def param_approx_b_opti(param_th, freq):
    """Create the Objective function for the dynamic compressibility beta.

    Arguments
    ---------
    param_th : list
        Physical parameters [ainf, M, N, L] for alpha and [gam, Mp, Np, Lp] for
        beta.
    freq : list
        Angular frequency w = 2 pi f

    Returns
    -------
    param_approx : callable object
        Objective function for the dynamic compressibility beta.
    """

    # Parameter definition
    nf = len(freq)
    jom = 2 * np.pi * 1j * freq
    th = np.zeros(nf,dtype=complex)

    gam, Mp, Np, Lp = param_th

    for i in range(nf):
        th[i] = gam - (gam - 1) / (
            1 + Mp / jom[i] + Np * (np.sqrt(1 + jom[i]/Lp) - 1) / jom[i]
        )

    def param_approx(param, err='mse', file_od='', th=th, jom=jom, freq=freq):
        """Objective function for the dynamic compressibility beta.

        Arguments
        ---------
        param : list
            [rk,sk] (MM parameters) [list of two arrays]
        err : str, default='mse'
            Choice of the error to return.
            - 'max': maximal absolute error [float],
            - 'mse': mean squared error [float].
        file_od : str, defaults=''
            Name of the file containing the mm parameter

        Returns
        -------
        error : float
            Error between the MM approximation and the JCAPL model
        """
        # Parameter definition
        nf = len(freq)

        mm = np.zeros(nf,dtype=complex)

        # Error
        if file_od:
            OD = OD_rep(file_od,0)
            mm = OD.fun(jom)
        else:
            mm += 1
            for k in range(len(param[0])):
                mm += param[0][k] / (jom - param[1][k])

        abs_err = np.abs(th-mm)
        if err=='max':
            error = np.max(abs_err)
        elif err=='mse':
            error = np.sum(abs_err*abs_err)/nf
        else:
            print("*** Wrong choice for the error choice ***")
            error = 0

        return error

    return param_approx
