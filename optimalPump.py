import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import coolingutils
import utils
from multiprocessing import Pool

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "16"
    })

"""
optimalPump.py
Purpose: Determine the pump that creates the longest lasting level under a certain correlation variance.
Date: 23 Sept 2024
Paul RB Hughes
"""

# nbc = 0
# nbm = 75
nbs = np.arange(0.1, 250.6, 5)
# scale = 1/(nbm + 1)
target = 1e-12
# tf = 0.08
gees = np.logspace(-1.5, 2.1, 1000)
tf = 100
# tfs = np.flip(np.logspace(-.8, 2, 1000))
colors = ['dodgerblue', 'navy']


def optimal(z, corrlevel):
    # feed cls largest to smallest
    # feed gees small to large
    gopt = np.zeros(np.size(corrlevel))
    minima = np.zeros(np.size(corrlevel))
    # tausarray = np.zeros([np.size(gees), np.size(corrlevel)])
    taumax = np.zeros(np.size(corrlevel))
    with Pool(processes=15) as pool:
        tausarray = pool.map(timeunder, range(np.size(gees)))
    for j in range(np.size(corrlevel)):  # I hate this...
        corrtimes = np.zeros(np.size(gees))
        for n in range(np.size(gees)):
            corrtimes[n] = tausarray[n][j]
        taumax[j] = np.max(corrtimes)
        gopt[j] = gees[np.argmax(corrtimes)]
        mingindex = np.nonzero(corrtimes)[0][0]
        minima[j] = gees[mingindex]
        # print(f"finished level {corrlevel[j]}")

    return gopt, taumax, minima


gfig, gax = plt.subplots()
# axins = inset_axes(gax, width="50%", height="75%",
#                    bbox_to_anchor=(.32, .05, .65, .55),
#                    bbox_transform=gax.transAxes, loc="lower left")
tfig, tax = plt.subplots()
levels = np.flip(np.logspace(-1, -0.01, 10000))
C, N = np.meshgrid(levels, nbs)
# zetas = [0.99, 1]
Gopts = np.zeros(np.shape(C))
Topts = np.zeros(np.shape(C))
zeta = 0.995
for i, nbm in enumerate(nbs):
    scale = 1/(nbm + 1)
    corrmax = 2 * scale


    def timeunder(k):
        g = gees[k]
        # tf = tfs[k]  # there is a tf we have previously determined that captures the end of entanglement and not much longer
        ic = np.array([-0.5 * zeta * scale * nbm, -scale * nbm, 0])  # we set the IC according to the tilde eqns
        t, state = utils.rel_simulation(zeta, g, 0, 0, ic, corrmax, target, tf)  # and simulate the squeeze
        corr = utils.rel_corr_var(state)  # get the correlation variance
        mc = np.min(corr)  # find its minimum
        taus = np.zeros(np.size(levels))
        # now I want to find time it is entangled below the targets i've set
        for j, cl in enumerate(levels):  # so for each target level
            if mc > (scale * cl):  # if the minimum correlation doesn't break the target
                break  # end the loop and use a stronger g
            entangledindex = np.nonzero(corr < scale * cl)  # if it does then find the indexes where the correlation does
            tau = t[entangledindex[0][-1]] - t[entangledindex[0][0]]  # so we can see how long it stays under
            taus[j] = tau
        print(f'g={g} done...')
        return taus


    opts = optimal(zeta, levels)
    Gopts[i, :] = opts[0]
    Topts[i, :] = opts[1]
    print(f"nb={nbm} done")

np.save(f"goptszeta{zeta}", Gopts)
np.save(f"toptszeta{zeta}", Topts)
np.save(f"goptszeta{zeta}NBS", N)
np.save(f"goptszeta{zeta}LEV", C)

GCF = gax.contourf(C, N, Gopts, cmap='viridis', origin="lower")
TCF = tax.contourf(C, N, Topts, cmap='viridis', origin="lower")
# gax.loglog(levels, opts[0], linewidth=2, label=zeta, color=colors[i], linestyle='-')
    # gbound = (1 + (1 - zeta)*nbm)*np.reciprocal(levels) - 1  # this is a lower bound
    # gmin = opts[2]
    # gax.loglog(levels, gmin, linewidth=2, color=colors[i], linestyle='--')
    # axins.loglog(levels, gmin, linewidth=2, color=colors[i], linestyle='--')
    # gax.loglog(levels, gbound, linewidth=2, color=colors[i], linestyle=':')
# tax.loglog(levels, opts[1], linewidth=2, label=zeta, color=colors[i], linestyle='-')
# np.savetxt(f"optPumps{zeta}", opts)

gax.set_xlabel(r"$\Delta^2_{t}$")
gax.set_ylabel(r"$n_m^b$")
gfig.colorbar(GCF, ax=gax, label=r"$g_{opt}$")


# gax.set_ylim([3e-2, gax.get_ylim()[1]])
#
# solid = mlines.Line2D([], [], color=colors[1], linestyle='-', label='$g_{opt}$')
# dashed = mlines.Line2D([], [], color=colors[1], linestyle='--', label='$g_{min}$')
# dotted = mlines.Line2D([], [], color=colors[1], linestyle=':', label='$g_{bound}$')
# first_legend = gax.legend(handles=[solid, dashed, dotted], loc=8)
# gax.add_artist(first_legend)
#
# gax.legend(title="$\zeta$",loc="lower left")

tax.set_xlabel(r"$\Delta^2_{t}$")
tax.set_ylabel(r"$n_m^b$")
tfig.colorbar(TCF, ax=tax, label=r"$\tilde\tau_{opt}$")
# tax.legend(title="$\zeta$")
#
# gax.tick_params(axis='y', direction='in', top=True, right=True, which='both')
# tax.tick_params(axis='y', direction='in', top=True, right=True, which='both')
# gax.tick_params(axis='x', direction='in', top=True, left=True, which='major')
# tax.tick_params(axis='x', direction='in', top=True, right=True, which='major')

plt.tight_layout()
# plt.show()
gfig.savefig(f"optimalPumpVaryBathZeta{zeta}.pdf", format='pdf', dpi=1200)
tfig.savefig(f"optimalTimeVaryBathZeta{zeta}.pdf", format='pdf', dpi=1200, bbox_inches='tight')
