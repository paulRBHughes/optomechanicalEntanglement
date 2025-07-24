import numpy as np
import matplotlib.pyplot as plt

import utils
from multiprocessing import Pool

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "16"
    })

"""
optimalPumpVaryBath.py
Purpose: Determine the pump that creates the longest lasting level under a certain correlation variance while varying nb. 
          Creates the data for figs 11&12 in manuscript
Modified to include nbm variations June 2025
Paul RB Hughes
"""

# nbs = np.arange(0.5, 250, 5)
target = 1e-11
gees = np.logspace(-2, 2.9, 500000)  # my setup can handle up to g=10^2.9, but not a hard limit
tf = 1000  # not too relevant with corrmax
levels = np.flip(np.logspace(-4, -0.003, 1000))  # target correlation levels to consider
# zetas = np.arange(0.99, 1, 0.002)
path = "pumpOptimizationData/paperData/intercepts/"
plotpath = "plots/"


def optimal():
    # feed cls largest to smallest
    # feed gees small to large
    # returns the optimal g value from the gees space for all levels, and the time they stay under the cv target.
    gopt = np.zeros(np.size(levels))
    taumax = np.zeros(np.size(levels))
    with Pool(processes=15) as pool:  # multiprocess parallelism, makes it go way faster
        tausarray = pool.map(timeunder, range(np.size(gees)))  # This function is (re)defined in the loop for nbs. Finding the time below all levels at each g
    for j in range(np.size(levels)):
        corrtimes = np.zeros(np.size(gees))
        for n in range(np.size(gees)):
            corrtimes[n] = tausarray[n][j]  # just some data structuring. Get the time under CV level for this level at each g
        taumax[j] = np.max(corrtimes)
        gopt[j] = gees[np.argmax(corrtimes)]
    return gopt, taumax


# C, N = np.meshgrid(levels, nbs)  # Creating a grid for data structure
# Gopts = np.zeros(np.shape(C))
# Topts = np.zeros(np.shape(C))
# for zeta in zetas:
# for i, nbm in enumerate(nbs):
# scale = 1/(nbm + 1)
corrmax = 1


def timeunder(k):
    # find the time below a certain correlation level for one g, indexed by k
    g = gees[k]
    ic = np.array([0, 0, 0])  # we set the IC according to the tilde eqns
    t, state = utils.rel_simulation(1, g, 0, 0, ic, corrmax, target, tf)  # and simulate the squeeze
    corr = utils.rel_corr_var(state)  # get the correlation variance
    mc = np.min(corr)  # find its minimum
    taus = np.zeros(np.size(levels))
    # now I want to find time it is entangled below the targets I've set
    for j, cl in enumerate(levels):  # so for each target level
        if mc > (cl):  # if the minimum correlation doesn't break the target
            break  # end the loop and use a stronger g
        entangledindex = np.nonzero(corr < cl)  # if it does then find the indexes where the correlation does
        tau = t[entangledindex[0][-1]] - t[entangledindex[0][0]]  # so we can see how long it stays under
        taus[j] = tau
    print(f'g={g} done...')
    return taus


opts = optimal()
np.save(path+f"Gints", opts[0])
np.save(path+f"Tints", opts[1])
    # Gopts[i, :] = opts[0]
    # Topts[i, :] = opts[1]
# print(f"zeta={zeta} done")
#
# np.save(path+f"Gints", Gopts)
# np.save(path+f"Tints", Topts)
# np.save(path+f"NBs", N)
# np.save(path+f"levels", C)

# GCF = gax.contourf(C, Z, Gopts, cmap='viridis', origin="lower")
# TCF = tax.contourf(C, Z, Topts, cmap='viridis', origin="lower")
# gax.loglog(levels, opts[0], linewidth=2, label=zeta, color=colors[i], linestyle='-')
    # gbound = (1 + (1 - zeta)*nbm)*np.reciprocal(levels) - 1  # this is a lower bound
    # gmin = opts[2]
    # gax.loglog(levels, gmin, linewidth=2, color=colors[i], linestyle='--')
    # axins.loglog(levels, gmin, linewidth=2, color=colors[i], linestyle='--')
    # gax.loglog(levels, gbound, linewidth=2, color=colors[i], linestyle=':')
# tax.loglog(levels, opts[1], linewidth=2, label=zeta, color=colors[i], linestyle='-')
# np.savetxt(f"optPumps{zeta}", opts)

# gax.set_xlabel(r"$\Delta^2_{t}$")
# gax.set_ylabel(r"$\zeta$")
# gfig.colorbar(GCF, ax=gax, label=r"$g_{opt}$")


# gax.set_ylim([3e-2, gax.get_ylim()[1]])
#
# solid = mlines.Line2D([], [], color=colors[1], linestyle='-', label='$g_{opt}$')
# dashed = mlines.Line2D([], [], color=colors[1], linestyle='--', label='$g_{min}$')
# dotted = mlines.Line2D([], [], color=colors[1], linestyle=':', label='$g_{bound}$')
# first_legend = gax.legend(handles=[solid, dashed, dotted], loc=8)
# gax.add_artist(first_legend)
#
# gax.legend(title="$\zeta$",loc="lower left")

# tax.set_xlabel(r"$\Delta^2_{t}$")
# tax.set_ylabel(r"$\zeta$")
# tfig.colorbar(TCF, ax=tax, label=r"$\tilde\tau_{opt}$")
# tax.legend(title="$\zeta$")
#
# gax.tick_params(axis='y', direction='in', top=True, right=True, which='both')
# tax.tick_params(axis='y', direction='in', top=True, right=True, which='both')
# gax.tick_params(axis='x', direction='in', top=True, left=True, which='major')
# tax.tick_params(axis='x', direction='in', top=True, right=True, which='major')

# plt.tight_layout()
# plt.show()
# gfig.savefig(f"optimalPumpVaryZetaCORNER.pdf", format='pdf', dpi=1200)
# tfig.savefig(f"optimalTimeVaryZetaCORNER.pdf", format='pdf', dpi=1200, bbox_inches='tight')
