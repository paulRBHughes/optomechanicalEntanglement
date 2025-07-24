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
slopes.py
Purpose: For a given optimal g, determine the ratio between the CV target and bath population in the large bath limit 
Date: 4 July 2025
No longer useful - Depricated
Paul RB Hughes
"""

savedata = True
path = 'ratios/zetatoone/nFourtyNine/'

zetas = np.arange(0.99, 0.99991, 0.0001)
# zetas = np.array([0.36])
gees = np.logspace(-2, 2.9, 1000)
levels = np.flip(np.logspace(-4.5, -0.3, 10000))
corrmax = 1
target = 1e-11
tf = 10000


def optimal():
    # feed cls largest to smallest
    # feed gees small to large
    gopt = np.zeros(np.size(levels))
    taumax = np.zeros(np.size(levels))
    with Pool(processes=15) as pool:
        tausarray = pool.map(timeunder, range(np.size(gees)))
    for j in range(np.size(levels)):  # I hate this...
        corrtimes = np.zeros(np.size(gees))
        for n in range(np.size(gees)):
            corrtimes[n] = tausarray[n][j]
        taumax[j] = np.max(corrtimes)
        gopt[j] = gees[np.argmax(corrtimes)]

    return gopt, taumax

for zeta in zetas:


    def timeunder(k):
        g = gees[k]
        ic = np.array([-0.5 * zeta * 0.98, -0.98, 0])  # we set the IC according to the tilde eqns
        t, state = utils.rel_simulation(zeta, g, 0, 0, ic, corrmax, target, tf)  # and simulate the squeeze
        corr = utils.rel_corr_var(state)  # get the correlation variance
        mc = np.min(corr)  # find its minimum
        taus = np.zeros(np.size(levels))
        # now I want to find time it is entangled below the targets I've set
        for j, cl in enumerate(levels):  # so for each target level
            if mc > cl:  # if the minimum correlation doesn't break the target
                break  # end the loop and use a stronger g
            entangledindex = np.nonzero(corr < cl)  # if it does then find the indexes where the correlation does
            tau = t[entangledindex[0][-1]] - t[entangledindex[0][0]]  # so we can see how long it stays under
            taus[j] = tau
        # print(f'g={g} done...')
        return taus


    gopt, taumax = optimal()
    np.save(path+f"Gopts{zeta}", gopt)
    np.save(path+f"Topts{zeta}", taumax)
    print(f"zeta = {zeta} done...")
np.save(path+f"targets", levels)

# fig, zx = plt.subplots(2)
# zx[0].loglog(levels, gopt)
# zx[1].semilogx(levels, taumax)
# plt.show()


