#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 21:24:16 2018

@author: mykhailo
"""
# %% read data
# from numpy import *
import numpy as np
from lmfit import Model
from lmfit.models import LinearModel
import matplotlib.pyplot as plt
import pandas as pd

folder = '/Users/mykhailo/Dropbox/URSULA/#002_364mT_374mT_step10uT_15mK_powerSweep_19-48-22/'

filepath_s21 = folder + 'vna02_power_set_magnet2_field_set_frequency_set.dat'
power, b_s21T, fHz, s21dB, s21deg = np.loadtxt(filepath_s21, unpack=True)

data_dB_filePath = folder + 's21dB'
data_freq_filePath = folder + 'freqGHz'
data_BmT_filePath = folder + 'BmT'


# %%
def phaser(x, y, fc, y0):

    def phaseFit(x, Qt, fr):
        return y0 - 180 * np.arctan(2 * Qt * (x - fr) / fr) / np.pi

    gmodel = Model(phaseFit)
    gmodel = gmodel + LinearModel()
    
    # print(gmodel.param_names)
    # print(gmodel.independent_vars)
    
    df = 0.002
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

    
# %% Curve fit
N = 1001  # number of VNA points for a single magnetic field value
res_phase = []
f_range = 10 * 10 ** 6  # 10[MHz] freq range
N_power_points = 3  # number of different power scanned
N_power_magnet = 1001  # number of magnetic field points for single power scan

for p in range(N_power_points):

    start = p * N_power_magnet
    end = (p + 1) * N_power_magnet

    if p == 0:
        label = '_n70dBm.txt'
    if p == 1:
        label = '_n60dBm.txt'
    if p == 2:
        label = '_n50dBm.txt'

    data_dB_file = open(data_dB_filePath + label, 'w')
    data_freq_file = open(data_freq_filePath + label, 'w')
    data_BmT_file = open(data_BmT_filePath + label, 'w')

    for b in range(start, end):
        
        x = fHz[b * N:(b + 1) * N]
        y_deg = s21deg[b * N:(b + 1) * N]
        y_degNew = []
        for y in y_deg:
            if y < -50:
                y = y + 360
    
            y_degNew.append(y)
        
        # data formatting for surface plot
        for ele in s21dB[b * N:(b + 1) * N]: 
            data_dB_file.write(str(ele) + ' ')  # in one raw for one magnetic field
        for freq in x / 10 ** 9:    
            data_freq_file.write(str(freq) + ' ')
        if b < N:
            for mT in b_s21T[b * N:(b + 1) * N]:
                data_BmT_file.write(str(mT * 1000) + ' ')
            
        data_dB_file.write('\n')
        data_freq_file.write('\n')
        data_BmT_file.write('\n')
    
        x = x[350:700]
        y_deg = y_degNew[350:700]
    
        df = pd.DataFrame(y_deg)
        # print(df.mean())
        
        y0 = float(df.mean())
        res_phase.append(phaser(x / 10 ** 9, y_deg, 7.368, y0))  # do not forget x/10**9
        
    data_dB_file.close()
    data_freq_file.close()
    data_BmT_file.close()
# %%
b = 3002
result = res_phase[b]

x = fHz[b * N:(b + 1) * N] / 10 ** 9
y_deg = s21deg[b * N:(b + 1) * N]

y_degNew = []
for y in y_deg:
   if y < -50:
      y = y + 360
   y_degNew.append(y)
                
x = x[350:700]
y_deg = y_degNew[350:700]
          
print(result.fit_report())
plt.plot(x, y_deg, 'bo')
plt.plot(x, result.init_fit, 'k--')
plt.plot(x, result.best_fit, 'r-')
plt.show()

# %%
import qcodes as qc

plot_phaseQt = qc.QtPlot()
plot_redchiPhase = qc.QtPlot()

for p in range(N_power_points):
    bT = []
    qt_phaseList = []
    redchiPhaseList = []
    
    for i in range(p * N_power_magnet, (p + 1) * N_power_magnet):
        qt_phaseList.append(res_phase[i].best_values['Qt'])
        redchiPhaseList.append(res_phase[i].redchi)
        bT.append(b_s21T[i * N])

    plot_phaseQt.add(bT, qt_phaseList)
    plot_redchiPhase.add(bT, redchiPhaseList)
    plt.plot(bT, qt_phaseList, 'bo')

    if p == 0:
        label = '_n70dBm.txt'
    if p == 1:
        label = '_n60dBm.txt'
    if p == 2:
        label = '_n50dBm.txt'

    fileQt = folder + 'Qtphase' + label
    fileRedChi = folder + 'QtredChi' + label
    f = open(fileQt, 'w')
    f2 = open(fileRedChi, 'w')
    for ele in qt_phaseList:
        f.write(str(ele) + '\n')   
    for chi in redchiPhaseList:
        f2.write(str(chi) + '\n')    
    f.close()
    f2.close()
