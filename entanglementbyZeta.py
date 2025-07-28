import numpy as np
import matplotlib.pyplot as plt
import coolingutils
import utils
from multiprocessing import Pool


"""
entanglementbyZeta.py
Purpose: to determine the time below the entanglement threshold for certain pumps as a function of zeta
Paul RB Hughes
"""

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "16"
    })

savefigs = False
path = "plots/entanglementTime.pdf"

gees = np.array([0.5, 2, 5])  # pumps consider
zetas = np.arange(0.98, 0.9999, 0.0001)  # zetas
nbm = 75
scale = 1/(nbm + 1)
maxcorr = 2 * scale
target = 1e-11
tf = 10


def largecooled(g):
    # determines the minimum correlation variance and time under the entanglement threshold for the pump strength
    mincorr = np.zeros(np.size(zetas))  # looking at things as a function of zeta
    enttime = np.zeros(np.size(zetas))
    for i, z in enumerate(zetas):
        ic = np.array([-0.5 * z * scale * nbm, -scale * nbm, 0])  # assumes large bath for mech
        t, state = utils.rel_simulation(z, g, 0, 0, ic, maxcorr, target, tf)
        corr = utils.rel_corr_var(state)
        mc = np.min(corr)
        mincorr[i] = mc
        # now I want to find time it is entangled
        if mc > scale:
            continue
        entangledindex = np.nonzero(corr < scale)
        enttime[i] = t[entangledindex[0][-1]] - t[entangledindex[0][0]]
    return mincorr, enttime

fig, ax = plt.subplots()
sts = ['-', '--', '-.']  # linestyles
colors = ['lightskyblue', 'dodgerblue', 'navy']  # and colours
for i, g in enumerate(gees):
    mc, et = largecooled(g)
    ax.plot(zetas, et, label=gees[i], linewidth=2, linestyle=sts[i], color=colors[i])
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$\tilde\tau$")
ax.tick_params(axis='y', direction='in', top=True, right=True, which='both')
ax.tick_params(axis='x', direction='in', top=True, right=True, which='both')
ax.legend(title="$g_b$")
plt.tight_layout()
if savefigs:
    plt.savefig(path, format='pdf', dpi=1200, bbox_inches='tight')
plt.show()

