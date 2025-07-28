import numpy as np
import matplotlib.pyplot as plt

"""
For plotting the results of optimalPump when varying nbm. 
"""

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "18"
    })

path = "pumpOptimizationData/paperData/"
savefigs = False
sz = [7, 10]

zetas = np.arange(0.99, 1, 0.002)
gfig, gax = plt.subplots(3, 2, figsize=sz)
plt.subplots_adjust(wspace=0.01, hspace=0.01)
tfig, tax = plt.subplots(3, 2, figsize=sz)
plt.subplots_adjust(wspace=0.01, hspace=0.01)
positions = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
levels = np.linspace(-1.5, 2.1, 9)  # this is the gees range for a colorbar
timelevels = np.logspace(-1.5, 1.6, 9)

N = np.load(path+f'NBs.npy')
C = np.load(path+f'levels.npy')

for i, zeta in enumerate(zetas):
    T = np.load(path+f'Topts{zeta}.npy')
    G = np.load(path+f'Gopts{zeta}.npy')
    pos = positions[i]

    logG = np.log10(G)
    logT = np.log10(T)

    GCF = gax[pos[0]][pos[1]].contourf(C, N, logG, levels, cmap='viridis', origin="lower")
    TCF = tax[pos[0]][pos[1]].contourf(C, N, logT, cmap='viridis', origin="lower")
    gax[pos[0]][pos[1]].text(0.25, 200, f'$\zeta$ = {zeta}', bbox=dict(alpha=0.8, ec='k', fc='white', boxstyle='square'), size='16')
    tax[pos[0]][pos[1]].text(0.25, 200, f'$\zeta$ = {zeta}', bbox=dict(alpha=0.8, ec='k', fc='white', boxstyle='square'), size='16')
    gax[pos[0]][pos[1]].set_xlabel(r"$\Delta^2_{t}$")
    gax[pos[0]][pos[1]].set_ylabel(r"$n_m^b$")
    tax[pos[0]][pos[1]].set_xlabel(r"$\Delta^2_{t}$")
    tax[pos[0]][pos[1]].set_ylabel(r"$n_m^b$")
    gax[pos[0]][pos[1]].get_xaxis().set_visible(pos[0])
    tax[pos[0]][pos[1]].get_xaxis().set_visible(pos[0])
    gax[pos[0]][pos[1]].get_yaxis().set_visible(not pos[1])
    tax[pos[0]][pos[1]].get_yaxis().set_visible(not pos[1])

gbar = gfig.colorbar(GCF, ax=gax).set_label(label=r"$\log_{10} g_{opt}$", size=16)
if savefigs:
    gfig.savefig(path+f"gopts.pdf", format='pdf', dpi=1200)

tfig.colorbar(TCF, ax=tax).set_label(label=r"$\log_{10} \tilde\tau_{max}$", size=16)
if savefigs:
    tfig.savefig(path+f"tauMaxes.pdf", format='pdf', dpi=1200)
plt.show()
