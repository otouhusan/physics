#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings

warnings.filterwarnings('error')

def main():
    fig = plt.figure()
    plt.title('harmonic oscillater')
    ani = animation.FuncAnimation(fig, plot, fargs = [10], interval=100)
    plt.show()

def plot(frame,time):
    time_develop(time)
    # plt.cla()
    plt.clf()
    plt.title('potential barrier')
    plt.xlim([-10,10])
    plt.ylim([-1,1.3])
    plt.xlabel('x')
    plt.plot(xs[1:-1], [V(x) for x in xs[1:-1]])
    plt.plot(xs[1:-1], psir[1:-1])
    plt.plot(xs[1:-1], psii[1:-1])
    im = plt.plot(xs[1:-1], [psir[i]**2+psii[i]**2 for i in range(1,Nbin+1)])
    plt.legend([r'$V={}(0<x)$'.format(Vheight), r'$Re(\psi)$', r'$Im(\psi)$', r'$|\psi|^2$'])

# hbar = 1
def gauss(amp, width, x0, p0, start, end, step):
    xs=np.arange(start,end,step,dtype=float)
    gaussr = np.array([amp*np.exp(-((x-x0)/width/2)**2)*np.cos(p0*x) for x in np.arange(start,end,step,dtype=float)])
    gaussi = np.array([amp*np.exp(-((x-x0)/width/2)**2)*np.sin(p0*x) for x in np.arange(start,end,step,dtype=float)])
    return (gaussr, gaussi)

Vheight = 50
amp, width, x0, p0 = np.pi**(-0.25), 0.4, -3, 10
start, end, dx = -10, 10, 0.05
Nbin = int((end-start)/dx)
t, dt, dtrec = 0, np.pi/3200, np.pi/4
trec = dtrec

xs = np.arange(start,end,dx,dtype=float)
xs = np.append(xs, xs[0])
xs = np.insert(xs,0,xs[-2])
g = gauss(amp, width, x0, p0, start, end, dx)
psir = g[0]
psir = np.append(psir,psir[0])
psir = np.insert(psir,0,psir[-2])
psii = g[1]
psii = np.append(psii,psii[0])
psii = np.insert(psii,0,psii[-2])

wpsir = np.zeros(Nbin+2)
wpsii = np.zeros(Nbin+2)

def time_develop(Ntime):
    global psir,psii,t,trec
    for i in range(Ntime):
        # print(i)
        for j in range(1,Nbin+1):
            wpsir[j] = psir[j] + dt*(-0.5*(psii[j+1]-2*psii[j]+psii[j-1])/(dx**2) + V(xs[j]) * psii[j])
        wpsir[0] = wpsir[Nbin]
        wpsir[Nbin+1] = wpsir[1]
        for j in range(1,Nbin+1):
            wpsii[j] = psii[j] - dt*(-0.5*(wpsir[j+1]-2*wpsir[j]+wpsir[j-1])/(dx**2) + V(xs[j]) * wpsir[j])
        wpsii[0] = wpsii[Nbin]
        wpsii[Nbin+1] = wpsii[1]
        psir = wpsir[:]
        psii = wpsii[:]
        t += dt
        # if((trec-dt<t)&(t<trec+dt)):
        #     trec += dtrec
        #     plt.plot(xs[1:-1], [psir[i]**2+psii[i]**2 for i in range(1,Nbin+1)])
        #     plt.plot(xs[1:-1], psir[1:-1])
        #     plt.plot(xs[1:-1], psii[1:-1])
        #     plt.show()

def V(x):
    if x<0:
        return 0
    else :
        return Vheight

if __name__ == '__main__' :
    main()
