"""
@author: Mykhailo Savytskyi
@email: m.savytskyi@unsw.edu.au
"""

from numpy import *
import numpy as np
from lmfit import Model
from lmfit.models import LinearModel
import matplotlib.pyplot as plt
import pandas as pd


def phaseFitS21(x, y, fc, y0, df=0.009, dy=50, Qt=30000, dQt=30000):
    #    if abs(y) > 90:
    #        y = y - sign(y)*180

    def phaseFit(x, Qt, fr):
        return y0 - 180 * np.arctan(2 * Qt * (x - fr) / fr) / pi

    gmodel = Model(phaseFit)
    gmodel = gmodel + LinearModel()

    # print(gmodel.param_names)
    # print(gmodel.independent_vars)

    params = gmodel.make_params()

    params.add('y0', value=y0, min=y0 - dy, max=y0 + dy)
    params.add('Qt', value=Qt, min=Qt - dQt, max=Qt + dQt)
    params.add('fr', value=fc, min=fc - df, max=fc + df)
    params.add('intercept', value=1, vary=True)
    params.add('slope', value=0, vary=True)

    results = gmodel.fit(y, params, x=x)

    return results


def phaseFitS11(x, y, fc, y0, df=0.009, dy=50, Qt=30000, dQt=30000):
    def phaseFit(x, Qc, Qi, fr):
        return y0 - 180 * np.atan(2 * ((Qi * Qc) / (Qi + Qc)) * (x - fr) / fr) / pi - 180 * np.atan(
                2 * ((Qc * Qi) / (Qi - Qc)) * ((x - fr) / fr)) / pi

    pass


def peakFits21():
    pass


def peakFitS11():
    pass


def circleFit():
    pass


def input(file, folder, type):
    filepath = folder + '/' + file
    fHz, s21dB, s21deg = np.loadtxt(filepath, unpack=True)
    return fHz, s21dB, s21deg


def output(file, folder):
    pass


def plotFit(input, results):
    pass
