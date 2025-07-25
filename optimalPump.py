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
      Cleaned up Jul 2025.
Paul RB Hughes
"""

loaddata = False  # runs sims if false
noProcesses = 16  # how many processes to run in parallel
savedata = True
datapath = "pumpOptimizationData/paperData/analytic/"
savefigs = False
plotpath = "plots/"
showbound = False  # show the lower bound on the plot
showmins = False  # show gmin on plot

colors = ['navy', 'dodgerblue']  # for plotting, one for each zeta
gfig, gax = plt.subplots()
tfig, tax = plt.subplots()

zetas = [1]
nbm = 0

if not loaddata:
    # nbc = 0  # not used, assuming cavity mode is much higher freq than mechanical
    scale = 1/(nbm + 1)  # for implementing the bath independent equations (allows for larger pump strengths
    corrmax = scale  # sim will stop when the (scaled) CV gets above this
    target = 1e-11
    tf = 30  # not really needed since corrmax will end it before the time cutoff
    gees = np.logspace(-1.5, 2.9, 50000)
    levels = np.flip(np.logspace(-2.4, -0.003, 10000))  # flipped so they go into function correctly


    def optimal():
        # feed cls largest to smallest
        # feed gees small to large
        gopt = np.zeros(np.size(levels))
        minima = np.zeros(np.size(levels))
        taumax = np.zeros(np.size(levels))
        with Pool(processes=noProcesses) as pool:
            tausarray = pool.map(timeunder, range(np.size(gees)))  # multiprocessing pool to compute in parallel
        for j in range(np.size(levels)):
            corrtimes = np.zeros(np.size(gees))
            for n in range(np.size(gees)):
                corrtimes[n] = tausarray[n][j]  # just manipulating to put things in the right place
            taumax[j] = np.max(corrtimes)
            gopt[j] = gees[np.argmax(corrtimes)]
            mingindex = np.nonzero(corrtimes)[0][0]
            minima[j] = gees[mingindex]

        return gopt, taumax, minima


for i, zeta in enumerate(zetas):

    if not loaddata:


        def timeunder(k):  # this needs to be redefined for different zetas so parallel can work
            g = gees[k]
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


        opts = optimal()
        gopts = opts[0]
        taumaxes = opts[1]
        gmins = opts[2]
        if savedata:
            np.save(datapath + f"gopts{zeta}", gopts)
            np.save(datapath + f"tmaxes{zeta}", taumaxes)
            np.save(datapath + f"gmins{zeta}", gmins)
            np.save(datapath + f"levels{zeta}", levels)
    else:
        gopts = np.load(datapath + f"gopts{zeta}.npy")
        taumaxes = np.load(datapath + f"tmaxes{zeta}.npy")
        gmins = np.load(datapath + f"gmins{zeta}.npy")
        levels = np.load(datapath + f"levels{zeta}.npy")

    gax.loglog(levels, gopts, linewidth=2, label=zeta, color=colors[i], linestyle='-')
    if showbound:
        gbound = (1 + (1 - zeta)*nbm)*np.reciprocal(levels) - 1  # this is a lower bound
        gax.loglog(levels, gbound, linewidth=2, color=colors[i], linestyle=':')
    if showmins:
        gax.loglog(levels, gmins, linewidth=2, color=colors[i], linestyle='--')
    tax.loglog(levels, taumaxes, linewidth=2, label=zeta, color=colors[i], linestyle='-')

# plot stuff
gax.set_xlabel(r"$\Delta^2_{t}$")
gax.set_ylabel(r"$g_{b}$")
gax.set_ylim([3e-2, gax.get_ylim()[1]])

solid = mlines.Line2D([], [], color=colors[1], linestyle='-', label='$g_{opt}$')
ghandles = [solid]
if showmins:
    dashed = mlines.Line2D([], [], color=colors[1], linestyle='--', label='$g_{min}$')
    ghandles.append(dashed)
if showbound:
    dotted = mlines.Line2D([], [], color=colors[1], linestyle=':', label='$g_{bound}$')
    ghandles.append(dotted)
if len(ghandles) > 1:
    first_legend = gax.legend(handles=ghandles, loc=8)
    gax.add_artist(first_legend)

if len(zetas) > 1:
    gax.legend(title="$\zeta$",loc="lower left")
    tax.legend(title="$\zeta$")

tax.set_xlabel(r"$\Delta^2_{t}$")
tax.set_ylabel(r"$\tilde{\tau}_{max}$")

gax.tick_params(axis='y', direction='in', top=True, right=True, which='both')
tax.tick_params(axis='y', direction='in', top=True, right=True, which='both')
gax.tick_params(axis='x', direction='in', top=True, left=True, which='major')
tax.tick_params(axis='x', direction='in', top=True, right=True, which='major')

plt.tight_layout()

if savefigs:
    gfig.savefig(plotpath + f"optimalPump.pdf", format='pdf', dpi=1200)
    tfig.savefig(plotpath + f"optimalTime.pdf", format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
