import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "16"
    })

levels = np.logspace(-1.5, 2.1, 8)  # this is the gees range
timelevels = np.logspace(-1.5, 1.6, 8)
zetas = np.arange(0.99, 1, 0.0005)  # important to change because I'm not going to figure out how to load this in yet
gfig, gax = plt.subplots()
gax.set_xlabel(r"$\Delta^2_{t}$")
gax.set_ylabel(r"$n_m^b$")
tfig, tax = plt.subplots()
tax.set_xlabel(r"$\Delta^2_{t}$")
tax.set_ylabel(r"$n_m^b$")
N = np.load(f'optimalFrames/NBs.npy')
C = np.load(f'optimalFrames/levels.npy')
empty = np.zeros(np.shape(C))
cfpumps = gax.contourf(C, N, empty, np.log10(levels))
cftimes = tax.contourf(C, N, empty, np.log10(timelevels))
zeta_text_pump = gax.text(0.8, 225, '')
zeta_text_time = tax.text(0.8, 225, '')


def animatePump(i):
    zeta = zetas[i]
    G = np.load(f'optimalFrames/Gopts{zeta}.npy')
    logG = np.log10(G)
    zeta_text_pump.set_text(f'$\zeta = {math.ceil(zeta * 10000)/10000}$')
    cfpumps = gax.contourf(C, N, logG, np.log10(levels))
    return cfpumps


anim = animation.FuncAnimation(gfig, animatePump, frames=np.size(zetas))
gfig.colorbar(cfpumps, ax=gax, label=r"$\log_{10} g_{opt}$")
plt.tight_layout()
anim.save("optimalFrames/pumpMovie.gif", writer='pillow', fps=4)


def animateTime(i):
    zeta = zetas[i]
    T = np.load(f'optimalFrames/Topts{zeta}.npy')
    logT = np.log10(T)
    zeta_text_time.set_text(f'$\zeta = {math.ceil(zeta * 10000)/10000}$')
    cftimes = tax.contourf(C, N, logT, np.log10(timelevels))
    return cftimes


anim = animation.FuncAnimation(tfig, animateTime, frames=np.size(zetas))
tfig.colorbar(cftimes, ax=tax, label=r"$\log_{10} \tilde\tau_{opt}$")
plt.tight_layout()
anim.save("optimalFrames/timeMovie.gif", writer='pillow', fps=4)
