import numpy as np
import os.path
import sys

"""
This module file creates a class containing the
parameters defining the socillatory-diffusive (OD)
representation of an irrationnal complex function.
"""

class OD_rep(object):
    """
    Class that manages the oscillatory-diffusive
    representation of a complex function
    """

    def __init__(self, filename="", physicalEntity=None, param=[], verbosity=0):
        """
        Constructor : define the different variables
        needed to build the discrete OD representation
        Inputs:
            filename: string which defines the paths to the file containing
                      the poles and weights written
            physicalEntity: string defining the associated zone of the OD_rep
            param: array containing the weights and poles from quadratures
            verbosity: integer defining the amount
                       of information showed (1,2,or 3)
        """

        self.verbosity      = verbosity
        self.physicalEntity = physicalEntity

        self.nbPoles        = 0
        self.nbDifPoles     = 0
        self.nbOscPoles     = 0
        self.nbRealPoles    = 0
        self.nbImagPoles    = 0
        self.nbRealOscPoles = 0
        self.nbCompOscPoles = 0  # Remark: half of all complex conjugate poles

        self.c_0     = 0
        self.c_inf   = 0
        self.poles   = np.zeros(0,dtype=complex)
        self.weights = np.zeros(0,dtype=complex)

        self.filename = ""
        if (len(filename)>0):
            self.buildFromFile(filename,verbosity)
            #self.check()
        else:
            self.buildFromParam(param)
            #self.check()
        return

    def buildFromFile(self, filename, verbosity=1):
        """
        Read a file that contains the description of the OD representation
        and store it in the variables of the class
        Inputs:
            filename: string containing the path to the
                      descritpion of the OD representation
        """

        if os.path.isfile(filename):
            self.filename = filename
            file = open(self.filename, 'r')
        else:
            print("The file "+filename+" doesn't exist !")
            sys.exit()

        idx = 0
        readmode = 0
        if verbosity: print('Reading %s' % file.name)
        line = 'a'

        while line:
            """
            Read each line of the .od file consecutively, and assign to
            list converted at the end into numpy arrays
            """

            line = file.readline()
            line = line.strip()

            if line.startswith('$'):
                if line == '$Coefficients':
                    readmode = 1
                elif line == '$nbPoles':
                    readmode = 2
                elif line == '$Weights':
                    readmode = 3
                    idx = 0
                elif line == '$Poles':
                    readmode = 4
                    idx = 0
                else:
                    readmode = 0

            elif readmode:
                columns = line.split()

                if readmode == 1:
                    self.c_inf = float(columns[0])
                    self.c_0 = float(columns[1])
                if readmode == 2:
                    self.nbDifPoles     = int(columns[0])
                    self.nbRealOscPoles = int(columns[1])
                    self.nbCompOscPoles = int(columns[2]) # Rmk in constructor

                    self.nbOscPoles  = self.nbRealOscPoles + \
                                       2 * self.nbCompOscPoles
                    self.nbRealPoles = self.nbDifPoles + self.nbRealOscPoles
                    self.nbImagPoles = self.nbOscPoles - self.nbRealOscPoles
                    self.nbPoles     = self.nbDifPoles + self.nbOscPoles

                    self.weights = np.zeros(self.nbPoles, dtype=complex)
                    self.poles   = np.zeros(self.nbPoles, dtype=complex)
                if readmode == 3 and self.nbPoles > 0:
                    self.weights[idx] = complex(float(columns[0]),
                                                float(columns[1]))
                    idx += 1
                if readmode == 4 and self.nbPoles > 0:
                    self.poles[idx] = complex(float(columns[0]),
                                              float(columns[1]))
                    idx += 1

        file.close()

        return

    def buildFromParam(self, odParam):
        """
        Read a file that contains the description of the OD representation
        and store it in the variables of the class
        Inputs:
            isoParam: float giving the residue of the isolated
            odParam: array containing weights and poles
                     of the od representation
        """

        self.c_inf = 1.0
        self.c_0 = odParam[-1]

        self.nbDifPoles     = len(odParam[0])
        self.nbRealOscPoles = 0
        self.nbCompOscPoles = 0

        self.nbOscPoles  = self.nbRealOscPoles + 2 * self.nbCompOscPoles

        self.nbRealPoles = self.nbDifPoles + self.nbRealOscPoles
        self.nbImagPoles = self.nbOscPoles - self.nbRealOscPoles
        self.nbPoles     = self.nbDifPoles + self.nbOscPoles

        self.weights = np.zeros(self.nbPoles, dtype=complex)
        self.poles   = np.zeros(self.nbPoles, dtype=complex)

        self.weights = odParam[0]
        self.poles   = odParam[1]

        return

    def get_properties(self):
        """
        Show a certain amount of informations about the
        OD representation, according to the verbosity choice.
        """

        print("-------OD representation properties-------")
        print(" ** Model: c_inf + (c_0 / s) + \sum [weights / (s - poles)])")
        print(" - c_inf: {}".format(self.c_inf))
        print(" - c_0:   {}".format(self.c_0))

        print("\n ** Diffusive part   : {0} real".format(self.nbDifPoles),
              "pole(s)/weight(s).")
        if (self.verbosity > 1):
            for i in range(self.nbDifPoles):
                print("r_k = {:.2e} |".format(np.real(self.weights[i])),
                      " s_k = {:.2e}".format(np.real(self.poles[i])))

        print("\n ** Oscillatory part : {0} real".format(self.nbRealOscPoles),
              "and {0} complex conj".format(self.nbCompOscPoles),
              "pole(s)/weight(s).")
        if (self.verbosity > 1):
            for i in range(self.nbDifPoles, self.nbDifPoles+self.nbRealOscPoles):
                print("r_k = {:.2e} |".format(np.real(self.weights[i])),
                      " s_k = {:.2e}".format(np.real(self.poles[i])))

            for i in range(self.nbDifPoles+self.nbRealOscPoles, self.nbPoles):
                print("r_k = {:.2e} ".format(np.real(self.weights[i])),
                      "+ j {:.2e} | ".format(np.imag(self.weights[i])),
                      "s_k = {:.2e} ".format(np.real(self.poles[i])),
                      "+ j {:.2e}".format(np.imag(self.poles[i])))

        print("------------------------------------------")
        return

    def check(self):
        """
        Check if the poles and weights are well defined and ordered
        according to the number of real/complex poles and weights
        given in the source file.
        """

        verif = False
        for i in range(self.nbDifPoles):
            if (np.imag(self.weights[i]) != 0): verif=True
            if (np.imag(self.poles[i]) != 0): verif=True
        if verif:
            print("\nA supposed real diffusive pole or weight has an imaginary part.",
                  "\n -- Exit --")
            sys.exit()


        for i in range(self.nbDifPoles, self.nbDifPoles+self.nbRealOscPoles):
            if (np.imag(self.weights[i]) != 0): verif=True
            if (np.imag(self.poles[i]) != 0): verif=True
        if verif:
            print("\nA supposed real oscillatory pole or weight has an imaginary part.",
                  "\n -- Exit --")
            sys.exit()

        for i in range(self.nbDifPoles+self.nbRealOscPoles, self.nbPoles, 2):
            if np.imag(self.weights[i]) == 0 and np.imag(self.poles[i]) == 0:
                print("\nA supposed imaginary oscillatory pole-weight pair has no imaginary part.",
                      "\n -- Exit --")
                sys.exit()
            if ((np.abs(np.imag(self.weights[i]) + np.imag(self.weights[i+1])) > 1e-12
             or  np.abs(np.real(self.weights[i]) - np.real(self.weights[i+1])) > 1e-12)
             or (np.abs(np.imag(self.poles[i]) + np.imag(self.poles[i+1])) > 1e-12
             or  np.abs(np.real(self.poles[i]) - np.real(self.poles[i+1])) > 1e-12)):
                print("\nA complexe oscillatory pole-weight pair is not complex conjugate.",
                      "\n -- Exit --")
                sys.exit()

        if not verif: print('Poles and weights checked\n')


    def fun(self, var):
        """
        Compute the complex values of the function defined
        by the poles and wieghts of its OD representation.
        Inputs:
            var: array of complex defining the complex domain
        Outputs:
            val: array of complex containing the values of
                 the function on the complex domain 'var'
        """

        val = 0 * var
        val += self.c_inf

        for i in range(self.nbPoles):
            val += self.weights[i] / (var - self.poles[i])

        if self.c_0:
            val += self.c_0 / var

        return val