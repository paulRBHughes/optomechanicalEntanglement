import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "12"
    })

# zeta = 0.999
zetas = np.arange(0.99, 1, 0.002)
gfig, gax = plt.subplots(2, 3)
# plt.tight_layout()
plt.subplots_adjust(wspace=0.01,hspace=0.01)
tfig, tax = plt.subplots(2, 3)
# plt.tight_layout()
plt.subplots_adjust(wspace=0.01,hspace=0.01)
positions = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
levels = np.logspace(-1.5, 2.1, 8)  # this is the gees range
timelevels = np.logspace(-1.5, 1.6, 8)

N = np.load(f'SMFrames/NBs.npy')
C = np.load(f'SMFrames/levels.npy')

for i, zeta in enumerate(zetas):
    T = np.load(f'SMFrames/Topts{zeta}.npy')
    G = np.load(f'SMFrames/Gopts{zeta}.npy')
    pos = positions[i]

    logG = np.log10(G)
    logT = np.log10(T)

    GCF = gax[pos[0]][pos[1]].contourf(C, N, logG, np.log10(levels), cmap='viridis', origin="lower")
    TCF = tax[pos[0]][pos[1]].contourf(C, N, logT, np.log10(timelevels), cmap='viridis', origin="lower")
    gax[pos[0]][pos[1]].text(0.25, 200, f'$\zeta$ = {zeta}', bbox=dict(alpha=0.8, ec='k', fc='white', boxstyle='square'), size='12')
    tax[pos[0]][pos[1]].text(0.25, 200, f'$\zeta$ = {zeta}', bbox=dict(alpha=0.8, ec='k', fc='white', boxstyle='square'), size='12')
    gax[pos[0]][pos[1]].set_xlabel(r"$\Delta^2_{t}$")
    gax[pos[0]][pos[1]].set_ylabel(r"$n_m^b$")
    tax[pos[0]][pos[1]].set_xlabel(r"$\Delta^2_{t}$")
    tax[pos[0]][pos[1]].set_ylabel(r"$n_m^b$")
    gax[pos[0]][pos[1]].get_xaxis().set_visible(pos[0])
    tax[pos[0]][pos[1]].get_xaxis().set_visible(pos[0])
    gax[pos[0]][pos[1]].get_yaxis().set_visible(not pos[1])
    tax[pos[0]][pos[1]].get_yaxis().set_visible(not pos[1])

gbar = gfig.colorbar(GCF, ax=gax).set_label(label=r"$\log_{10} g_{opt}$", size=14)
# gfig.supxlabel(r"$\Delta^2_{t}$")
# gfig.supylabel(r"$n_m^b$")
gfig.savefig(f"SMFrames/optimalGs.pdf", format='pdf', dpi=1200)

tfig.colorbar(TCF, ax=tax).set_label(label=r"$\log_{10} \tilde\tau_{max}$")
# tfig.supxlabel(r"$\Delta^2_{t}$")
# tfig.supylabel(r"$n_m^b$")
tfig.savefig(f"SMFrames/tauMaxes.pdf", format='pdf', dpi=1200)
# plt.show()
