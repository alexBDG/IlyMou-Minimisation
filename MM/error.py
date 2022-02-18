#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: imoufid
"""

import numpy as np
from .OD_rep import OD_rep

def error_a(mm_param,phy_param,reference_val,freq,err='mse',file_od=''):
    """
    Objective function for the dynamic tortuosity alpha

    Input variables:
        . mm_param  = [rk,sk] (MM parameters) [list of two arrays],
        . phy_param = [O, M, N, L] (Physical parameters) [list of floats]:
            [ainf, M, N, L] for alpha and [gam, Mp, Np, Lp] for beta
        . reference_val (Theoretical results) [array of floats],
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

    ainf, M, N, L = phy_param

    # %% Error
    if file_od:
        OD = OD_rep(file_od,0)
        mm = OD.fun(jom)
    else:
        mm += ainf
        mm += ainf * M  / jom
        for k in range(len(mm_param[0])): mm += mm_param[0][k] / (jom - mm_param[1][k])

    th = reference_val
    abs_err = np.abs(th-mm)
    if err=='max':
        error = np.max(abs_err)
    elif err=='mse':
        error = np.sum(abs_err*abs_err)/nf
    else:
        print("*** Wrong choice for the error choice ***")
        error = 0

    return error

def error_b(mm_param,phy_param,reference_val,freq,err='mse',file_od=''):
    """
    Objective function for the dynamic compressibility beta

    Input variables:
        . mm_param  = [rk,sk] (MM parameters) [list of two arrays],
        . phy_param = [O, M, N, L] (Physical parameters) [list of floats]:
            [ainf, M, N, L] for alpha and [gam, Mp, Np, Lp] for beta
        . reference_val (Theoretical results) [array of floats],
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

    ainf, M, N, L = phy_param

    # %% Error
    if file_od:
        OD = OD_rep(file_od,0)
        mm = OD.fun(jom)
    else:
        mm += 1
        for k in range(len(mm_param[0])): mm += mm_param[0][k] / (jom - mm_param[1][k])

    th = reference_val
    abs_err = np.abs(th-mm)
    if err=='max':
        error = np.max(abs_err)
    elif err=='mse':
        error = np.sum(abs_err*abs_err)/nf
    else:
        print("*** Wrong choice for the error choice ***")
        error = 0

    return error

def error_a_opti(param_th, freq):
    """Create the Objective function for the dynamic tortuosity alpha.

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
        Objective function for the dynamic tortuosity alpha.
    """

    # model parameters
    ainf, M, N, L = param_th

    # range of frequencies
    jom = 2 * np.pi * 1j * freq
    Nf = len(jom)

    # Exact solutions
    tha = np.zeros(Nf,dtype=complex)
    for i in range(Nf):
        tha[i] = ainf * (1 + M / jom[i] + N * (np.sqrt(1 + jom[i] / L) - 1) / jom[i])

    ainf, M, N, L = param_th

    # %% Err function
    def err_a(mm_param,phy_param=[ainf, M, N, L],reference_val=tha,
              jom=jom,err='mse',file_od=''):
        """Objective function for the dynamic tortuosity alpha.

        Arguments
        ---------
        mm_param: list
            [rk,sk] (MM parameters) [list of two arrays]
        err: str, default='mse'
            Choice of the error to return.
            - 'max': maximal absolute error [float],
            - 'mse': mean squared error [float].
        file_od: str, defaults=''
            Name of the file containing the mm parameter

        Returns
        -------
        error : float
            Error between the MM approximation and the JCAPL model
        """

        nf = len(jom)
        ainf, M, N, L = phy_param
        mm = np.zeros(nf,dtype=complex)

        if file_od:
            OD = OD_rep(file_od,0)
            mm = OD.fun(jom)
        else:
            mm += ainf
            mm += ainf * M  / jom
            for k in range(len(mm_param[0])): mm += mm_param[0][k] / (jom - mm_param[1][k])

        th = reference_val
        abs_err = np.abs(th-mm)
        if err=='max':
            error = np.max(abs_err)
        elif err=='mse':
            error = np.sum(abs_err*abs_err)/nf
        else:
            print("*** Wrong choice for the error choice ***")
            error = 0

        return error

    return err_a

def error_b_opti(param_th, freq):
    """Create the Objective function for the dynamic tortuosity alpha.

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
        Objective function for the dynamic tortuosity alpha.
    """

    # model parameters
    gam, Mp, Np, Lp = param_th

    # range of frequencies
    jom = 2 * np.pi * 1j * freq
    Nf = len(jom)

    # Exact solutions
    thb = np.zeros(Nf,dtype=complex)
    for i in range(Nf):
        thb[i] = gam - (gam - 1) / (1 + Mp / jom[i] + Np * (np.sqrt(1 + jom[i]/Lp) - 1) / jom[i])

    ainf, M, N, L = param_th

    # %% Err function
    def err_b(mm_param,phy_param=[gam, Mp, Np, Lp],reference_val=thb,
              jom=jom,err='mse',file_od=''):
        """Objective function for the dynamic tortuosity alpha.

        Arguments
        ---------
        mm_param: list
            [rk,sk] (MM parameters) [list of two arrays]
        err: str, default='mse'
            Choice of the error to return.
            - 'max': maximal absolute error [float],
            - 'mse': mean squared error [float].
        file_od: str, defaults=''
            Name of the file containing the mm parameter

        Returns
        -------
        error : float
            Error between the MM approximation and the JCAPL model
        """

        nf = len(jom)
        mm = np.zeros(nf,dtype=complex)

        if file_od:
            OD = OD_rep(file_od,0)
            mm = OD.fun(jom)
        else:
            mm += 1
            for k in range(len(mm_param[0])): mm += mm_param[0][k] / (jom - mm_param[1][k])

        th = reference_val
        abs_err = np.abs(th-mm)
        if err=='max':
            error = np.max(abs_err)
        elif err=='mse':
            error = np.sum(abs_err*abs_err)/nf
        else:
            print("*** Wrong choice for the error choice ***")
            error = 0

        return error

    return err_b
