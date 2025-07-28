import numpy as np
import matplotlib.pyplot as plt
import coolingutils
import utils
from multiprocessing import Pool


"""
varianceMinima.py
Purpose: to determine the correlation variance minimum for a given pump
        Here I will vary the relative loss rates for our defined pump strengths and populations
Date: 3 Sept 2024
Produces Fig 6 of manuscript
Paul RB Hughes
"""

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "16"
    })

save = False  # for figures
path = "plots/entanglementStrength.pdf"

nbm = 75
scale = 1/(nbm + 1)
target = 1e-11
tf = 10  # this shouldn't be important, as sims stop when they reach cvmax
cvmax = 2 * scale
gees = np.arange(0.5, 5.1, 0.1)  # range of pump strengths
zetas = [0.99, 0.999, 1]
sts = ['--', '-.', '-']  # linestyles for the zetas
colors = ['lightskyblue', 'dodgerblue', 'navy']  # plot line colours for each zeta


def largecooled(zeta):
    # this function will return the minimum correlation variance and time below 1 for the pump strengths in gees
    # also assumes cooled down to the red sideband limit
    mincorr = np.zeros(np.size(gees))
    enttime = np.zeros(np.size(gees))
    for i, g in enumerate(gees):
        ic = np.array([-0.5 * zeta * scale * nbm, -scale * nbm, 0])  # assumes large bath for mech
        t, state = utils.rel_simulation(zeta, g, 0, 0, ic, cvmax, target, tf)
        corr = utils.rel_corr_var(state)
        mc = np.min(corr)
        mincorr[i] = mc
        # now I want to find time it is entangled. With the scaling, that is CV < 1/(nbm + 1)
        if mc > scale:
            continue
        entangledindex = np.nonzero(corr < scale)
        enttime[i] = t[entangledindex[0][-1]] - t[entangledindex[0][0]]
    return mincorr, enttime


fig, ax = plt.subplots()
for i, z in enumerate(zetas):
    mc, et = largecooled(z)
    ax.plot(gees, mc/scale, label=z, linewidth=2, linestyle=sts[i], color=colors[i])
    ax.set_xlabel(r"$g_b$")
    ax.set_ylabel(r"$\Delta_{12,min}^2$")
ax.legend(title="$\zeta$")
plt.tight_layout()
if save:
    plt.savefig(path, format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
