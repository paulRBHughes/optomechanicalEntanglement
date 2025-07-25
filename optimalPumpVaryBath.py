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

nbs = np.arange(5, 250, 5)
target = 1e-11
gees = np.logspace(-2, 2.9, 1000)  # my setup can handle up to g=10^2.9, but not a hard limit
tf = 100  # not too relevant with corrmax
levels = np.flip(np.logspace(-1, -0.03, 1000))  # target correlation levels to consider
zetas = np.arange(0.99, 1, 0.002)
path = "pumpOptimizationData/tester/"


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


C, N = np.meshgrid(levels, nbs)  # Creating a grid for data structure
Gopts = np.zeros(np.shape(C))
Topts = np.zeros(np.shape(C))
for zeta in zetas:
    for i, nbm in enumerate(nbs):
        scale = 1/(nbm + 1)
        corrmax = scale


        def timeunder(k):
            # find the time below a certain correlation level for one g, indexed by k
            g = gees[k]
            ic = np.array([-0.5 * zeta * scale * nbm, -scale * nbm, 0])  # we set the IC according to the tilde eqns
            t, state = utils.rel_simulation(zeta, g, 0, 0, ic, corrmax, target, tf)  # and simulate the squeeze
            corr = utils.rel_corr_var(state)  # get the correlation variance
            mc = np.min(corr)  # find its minimum
            taus = np.zeros(np.size(levels))
            # now I want to find time it is entangled below the targets I've set
            for j, cl in enumerate(levels):  # so for each target level
                if mc > (cl * scale):  # if the minimum correlation doesn't break the target
                    break  # end the loop and use a stronger g
                entangledindex = np.nonzero(corr < cl * scale)  # if it does then find the indexes where the correlation does
                tau = t[entangledindex[0][-1]] - t[entangledindex[0][0]]  # so we can see how long it stays under
                taus[j] = tau
            # print(f'g={g} done...')
            return taus


        opts = optimal()
        Gopts[i, :] = opts[0]
        Topts[i, :] = opts[1]
        print(f"nb = {nbm} done")
    np.save(path+f"Gopts{zeta}", Gopts)
    np.save(path+f"Topts{zeta}", Topts)
    print(f"zeta={zeta} done")

np.save(path+f"NBs", N)
np.save(path+f"levels", C)
