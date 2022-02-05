import numpy as np
import sys

"""
This module file initialize the parameters for
the definition of the dynamic variables.
"""

def init(case):
    """
    Input:
        case: integer
    Outputs:
        Parameters M, N, L (for alpha) an Mp, Np, Lp (for beta)
    """

    if case == 0: # JCAL (Alomar data)

        phi   = 0.99
        ainf  = 1.0
        k0    = 4.0e-9
        k0p   = 4.0e-9
        lamb  = 130e-6
        lambp = 160e-6

        rho   = 1.225
        sigma = 4500
        mu    = k0 * sigma
        nu    = mu / rho

        # 300 K
        Cp    = 1.005e3
        gam   = 1.403
        kappa = 26.24e-3
        Pr    = mu * Cp / kappa

        M  = nu * phi / (k0 * ainf)
        N  = nu * phi / (k0 * ainf)
        L  = nu * phi * phi * lamb * lamb / (4 * k0 * k0 * ainf * ainf)

        Mp  = nu * phi / (k0p * Pr)
        Np  = nu * phi / (k0p * Pr)
        Lp  = nu * phi * phi * lambp * lambp / (4 * k0p * k0p * Pr)

        a0 = 1e12
        a0p = 1e12

    elif case == 1: #JCAPL
        rho = 1.225
        mu  = 1.802e-5

        # 300 K
        Cp    = 1.005e3
        kappa = 2.624e-2
        Pr    = mu * Cp / kappa

        phi   = 0.99
        ainf  = 1.14
        k0    = 5.97e-9
        lamb  = 230e-6
        lambp = 1.33 * lamb
        k0p   = 2.38 * k0
        a0    = 1.45
        a0p   = 1.5

        St = lamb * lamb * rho / mu
        q  = (1. / (a0 - ainf)) * (2. * k0 * ainf * ainf) / (phi * lamb * lamb)
        M  = 8 * k0 * ainf / (phi * lamb * lamb)
        L  = 16 * (q * q) / (M * M * St)
        N  = 8 * q / (M * St)
        M  = 8 / (M * St)

        Stp = lambp * lambp * Pr * rho / mu
        qp = (1. / (a0p - 1)) * (2. * k0p) / (phi * lambp * lambp)
        Mp  = 8 * k0p / (phi * lambp * lambp)
        Lp  = 16 * (qp * qp) / (Mp * Mp * Stp)
        Np  = 8 * qp / (Mp * Stp)
        Mp  = 8 / (Mp * Stp)

    elif case == 2: # Horoshenkov
        phi   = 0.99
        ainf  = 1.0
        k0    = 4.0e-9
        k0p   = 4.0e-9
        lamb  = 130e-6
        lambp = 160e-6

        rho   = 1.225
        sigma = 4500
        mu    = k0 * sigma
        nu    = mu / rho

        # 300 K
        Cp    = 1.005e3
        kappa = 26.24e-3
        Pr    = mu * Cp / kappa

        # (Horoshenkov data)
        sigma_s = 0.3
        theta1 = 1/3
        theta2 = np.exp(-0.5*(sigma_s*np.log(2))**2)/np.sqrt(2)
        theta3 = theta1 / theta2

        M  = nu * phi / (k0 * ainf)
        N  = theta1
        L  = M / theta3

        theta1 = 1/3
        theta2 = np.exp(1.5*(sigma_s*np.log(2))**2)/np.sqrt(2)
        theta3 = theta1 / theta2

        Mp  = nu * phi / (k0p * Pr)
        Np  = theta1
        Lp  = Mp / theta3

    elif case == 3: # Horoshenkov (Alomar physical data)
        phi   = 0.99
        ainf  = 1.0
        k0    = 4.0e-9
        k0p   = 4.0e-9
        lamb  = 130e-6
        lambp = 160e-6

        rho   = 1.225
        sigma = 4500
        mu    = k0 * sigma
        nu    = mu / rho

        # 300 K
        Cp    = 1.005e3
        kappa = 26.24e-3
        Pr    = mu * Cp / kappa

        # (Arbitrary data)
        tau_v = 1 #alpha
        tau_e = 1 #beta

        M = 2/tau_v
        N = 1/tau_v
        L = 1/tau_v

        Mp = 2/tau_e
        Np = 1/tau_e
        Lp = 1/tau_e

    elif case==4: #IFAR
        r   = 0.5*1.04e-3
        #d  = 0.86e-3
        phi = 0.11

        # Phisycal quantities 300K
        rho   = 1.177
        mu    = 1.85e-5
        nu    = mu / rho
        sigma = 8 * mu / (phi * r*r)

        Cp    = 1.005e3
        gam   = 1.403
        kappa = 26.24e-3
        Pr    = mu * Cp / kappa

        lamb  = r
        lambp = r

        k0    = mu / sigma
        k0p   = k0
        ainf  = 1.0
        M  = nu * phi / (k0 * ainf)
        N  = nu * phi / (k0 * ainf)
        L  = nu * phi * phi * lamb * lamb / (4 * k0 * k0 * ainf * ainf)

        Mp  = nu * phi / (k0p * Pr)
        Np  = nu * phi / (k0p * Pr)
        Lp  = nu * phi * phi * lambp * lambp / (4 * k0p * k0p * Pr)

    else:
        print('\n Wrong choice for case\n')
        sys.exit()

    return M,N,L,Mp,Np,Lp,ainf,gam
