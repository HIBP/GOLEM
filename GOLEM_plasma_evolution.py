# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 19:39:19 2022

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patch

#%%    Names and adresses / Адреса и имена

Cache_adress = 'F:\\cache\\GOLEM\\'
GOLEM_SIGNALS = {'I_pl': 'U_IntRogCoil', 
                 # Plasma current / Ток плазмы
                 'U_loop': 'U_Loop',                  
                 # Loop voltage / Напряжение на обходе
                 'B_t': 'U_IntBtCoil',                
                 # Toroidal magnetic field / Магнитное торполе
                 'MC16': 'ring_',                     
                 # MHD Mirnov coils / МГД магнитные зонды
                 'MC_lim': 'U_mc',                    
                 # Limiter Mirnov coils / Магнитные зонды за лимитером
                 'LP': 'TektrMSO56_LP',               
                 # Langmuir probe / Ленгмюровский зонд
                 'Photodiode': 'U_LeybPhot'           
                 # AXUV / Болометр
                 }

GOLEM_CALIB = {}

#%%    Constants / Константы

R_0    = 40                # Major radius [cm] / Большой радиус (см)
r_0    = 10                # Minor radius [cm] / Малый радиус (см)
r_lim  = 8.5               # Limiter radius [cm] / Радиус лимитера (см)
r_MC   = 9.3               # Mirnov coils radius [cm] / Радиус расположения магнитных зондов (см)
Aeff16   = [68.93, 140.68, 138.83, 140.43,
            68.59, 134.47, 134.28, 142.46,
            67.62, 142.80, 140.43, 140.43,
            67.62, 140.68, 139.82, 139.33]     # Area of MC16 [cm^2] / Эффективная площадь магнитных зондов набора MC16 (см^2)

for i in range(len(Aeff16)):
    GOLEM_CALIB['ring_'+ str(i+1)] = 1.0/Aeff16[i]

#%%    Resemple

def pack_sig(t, y): 
    N = len(t)
    result = np.zeros((2, N), np.double)
    result[0, :] = t
    result[1, :] = y
    return result

def sig_resample_as(sig, refsig): 
    ref_x = refsig[0]
    new_y = np.interp(ref_x, sig[0], sig[1])
    return pack_sig(ref_x, new_y)

def sig_fragment(sig, timerange):
    t = sig[0]
    y = sig[1]
    j0 = np.searchsorted(t, timerange[0])
    j1 = np.searchsorted(t, timerange[1])
    return np.vstack( ( t[j0:j1],  y[j0:j1] )  )

#%%    Calibrate zero level / Калибровка нуля

def calib_sig(t, y, timerange = None): 
    if timerange is None: 
        #timerange = (0.2, 0.9)
        timerange = (0.37, 2.4)
    j0 = np.searchsorted(t, timerange[0])
    j1 = np.searchsorted(t, timerange[1])
    zerolevel = np.mean(y[j0:j1])
    return y - zerolevel

#%%    Integrate / Интегратор

def integrate_sig(t, y, calib_by = None): 
    y = calib_sig(t, y, calib_by)
    dt = (t[-1]-t[0])/(len(t)-1)
    return np.cumsum(y)*dt

#%%    Read from file / Чтение из файла

def read_signal (shot, signal_name, channel = None):
    chan = '' if channel is None else str(channel)
    internal_signal_name = GOLEM_SIGNALS[signal_name]
    internal_signal_name += chan
    calib_coef = GOLEM_CALIB.get(internal_signal_name, 1.0)
    fname = Cache_adress + str(shot) + '\\' + internal_signal_name + '.csv'
    sig = np.loadtxt(fname = fname, dtype = float, comments = '#', delimiter = ',', 
                                usecols = (0,1) )
    sig[:,0] *= 1000.0
    sig[:,1] *= calib_coef
    return sig.T

#%%   Mirnov coils for plasma position / Магнитные зонды для положения плазмы

def MC_for_position(shot):
    
    MC1 = read_signal(shot, 'MC16', 1)
    MC5 = read_signal(shot, 'MC16', 5)
    MC9 = read_signal(shot, 'MC16', 9)
    MC13 = read_signal(shot, 'MC16', 13)
    MC_for_position = np.vstack( (MC1[0], MC1[1], MC5[1], MC9[1], MC13[1]) )
    
    return MC_for_position

#%%    Plasma position and radius / Положение плазмы и радиус

def plasma_position(shot):
    MC = MC_for_position(shot)
    
    B_1  =  integrate_sig(MC[0], MC[1])    # T
    B_5  =  integrate_sig(MC[0], MC[2])    # T
    B_9  =  integrate_sig(MC[0], MC[3])    # T
    B_13 =  integrate_sig(MC[0], MC[4])   # T
    Δz   =  r_MC*(B_5-B_13)/(B_5+B_13)         # cm
    Δr   =  r_MC*(B_1-B_9)/(B_1+B_9)           # cm
    r_pl = r_lim - (Δz**2 + Δr**2)**(0.5)      # cm
    r_pl[np.abs(r_pl) > 2*r_lim] = 0.0
    r_pl[r_pl < 0.0] = 0.0
    
    Δz   = pack_sig(MC[0], Δz)
    Δr   = pack_sig(MC[0], Δr)
    r_pl = pack_sig(MC[0], r_pl)
    
    return Δz, Δr, r_pl

#%%   Discharge duration / Длительность разряда

def discharge_duration(shot):
    Δz, Δr, r_pl = plasma_position(shot)
    
    for i in range(len(r_pl[0])):
        if r_pl[1, i] == 0 and r_pl[1, i+1] > 0:
            t_0 = r_pl[0, i]
            break
    for i in range(len(r_pl[0])):
        if r_pl[1, i] == 0 and r_pl[1, i-1] > 0 and r_pl[0, i] > t_0:
            t_1 = r_pl[0, i]   
            
    return t_0, t_1

#%%   Edge safety factor / q на границе

def edge_safety_factor(shot, I_pl, B_t):
    Δz, Δr, r_pl = plasma_position(shot)
    t_0, t_1 = discharge_duration(shot)
        
    q = ((5*r_pl[1]**2*B_t[1]*10.0)/((Δr[1] + R_0)*I_pl[1]*0.001)) 
    
    return pack_sig(r_pl[0], q)

#%%    Programm / Программа

shot   = 36598             # shot No. / Номер импульса

Δz, Δr, r_pl = plasma_position(shot)
t_0, t_1 = discharge_duration(shot)
timerange = (t_0, t_1)

I_pl = read_signal(shot, 'I_pl')
I_pl = sig_fragment(I_pl, timerange)
I_pl = sig_resample_as(I_pl, r_pl)

B_t = read_signal(shot, 'B_t')
B_t = sig_fragment(B_t, timerange)
B_t = sig_resample_as(B_t, r_pl)

q = edge_safety_factor(shot, I_pl, B_t)


#%%    Plots / Графики

f1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

plt.sca(ax1)
plt.plot(Δr[0], Δr[1], label='Horizontal shift', color='green')
plt.plot(Δz[0], Δz[1], label='Vertical shift', color='blue')
plt.plot(Δr[0], Δr[0]-Δr[0]+r_lim, color='black')
plt.plot(Δr[0], Δr[0]-Δr[0]-r_lim, color='black' )
plt.ylim(-r_0, r_0)
plt.xlim(t_0, t_1)
plt.ylabel('Radius, cm')
plt.legend(loc = 'lower right')

plt.sca(ax2)
plt.plot(r_pl[0], r_pl[1], label='Plasma radius', color='red')
plt.plot(Δr[0], Δr[0]-Δr[0]+r_lim, color='black')
plt.ylim(0, r_0)
plt.ylabel('Radius, cm')
plt.legend(loc = 'upper right')

plt.sca(ax3)
plt.plot(q[0], q[1], label='Edge safety factor', color='blue')
plt.ylabel('q, a.u.')
plt.legend(loc = 'upper right')

plt.sca(ax4)
plt.plot(I_pl[0], I_pl[1]*0.001, label='Plasma current', color='red')
plt.ylabel('Plasma current, kA')
plt.xlabel('time, ms')
plt.legend(loc = 'upper right')

#%% Plasma position plotter / Отрисовка положения плазмы
def updateGraph():
    global slider_time
    global graph_axes
    t = slider_time.val
        
    idx = np.searchsorted(r_pl[0], t)
    circ1_center = (R_0+Δr[1, idx], Δz[1, idx])
    circ1_radius = r_pl[1, idx]

    circ1 = patch.Circle( circ1_center, circ1_radius, color='r')
    circ2 = patch.Circle( (R_0,0), r_lim, edgecolor='black', fill = False, linewidth = 5)
    circ3 = patch.Circle( (R_0,0), r_0, edgecolor='black', fill = False, linewidth = 5)
        
    graph_axes.clear() # returns the axes to (0, 1)
    graph_axes.grid(axis = 'both', color = 'blue', linewidth = 0.5, linestyle = '--')
    graph_axes.set_xlim(left = R_0-r_0, right = R_0+r_0)
    graph_axes.set_ylim(bottom =-r_0, top =r_0)
    graph_axes.set(xlim=(R_0-r_0, R_0+r_0), ylim=(-r_0, r_0))
    graph_axes.add_patch(circ1)
    graph_axes.add_patch(circ2)
    graph_axes.add_patch(circ3)
    graph_axes.set_xlabel('r, cm')
    graph_axes.set_ylabel('z, cm')
    plt.draw()
        
def onChangeValue(value):
    updateGraph()

    # Создадим окно с графиком
fig2, graph_axes = plt.subplots(figsize=(6, 9))

    # Оставим снизу от графика место для виджетов
fig2.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.4)

    # Создание слайдера
axes_slider_time = plt.axes([0.15, 0.15, 0.5, 0.04])
slider_time = Slider(axes_slider_time,
                    label='time, ms',
                    valmin=discharge_duration(shot)[0],
                    valmax=discharge_duration(shot)[1],
                    valinit=discharge_duration(shot)[0],
                    valfmt='%1.2f',
                    valstep=0.1)

    #Подпишемся на событие при изменении значения слайдера.
slider_time.on_changed(onChangeValue)

    #updateGraph()
plt.show()