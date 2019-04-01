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


def phaser(x, y, fc, y0):
    #    if abs(y) > 90:
    #        y = y - sign(y)*180

    def phaseFit(x, Qt, fr):
        return y0 - 180 * np.arctan(2 * Qt * (x - fr) / fr) / pi

    gmodel = Model(phaseFit)
    gmodel = gmodel + LinearModel()

    # print(gmodel.param_names)
    # print(gmodel.independent_vars)

    df = 0.009
    dy = 50
    Qt = 30000
    dQt = Qt

    params = gmodel.make_params()

    params.add('y0', value=y0, min=y0 - dy, max=y0 + dy)
    params.add('Qt', value=Qt, min=Qt - dQt, max=Qt + dQt)
    params.add('fr', value=fc, min=fc - df, max=fc + df)
    params.add('intercept', value=1, vary=True)
    params.add('slope', value=0, vary=True)

    result = gmodel.fit(y, params, x=x)

    return result

