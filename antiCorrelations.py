# Purpose: to determine the anti-correlation variance of the optimal pump

import numpy as np
import matplotlib.pyplot as plt

import utils
from multiprocessing import Pool

# to begin, I need to load in the optimal values and corresponding nb, levels
zetas = np.arange(0.8, 1, 0.05)
levels = np.flip(np.logspace(-0.07, 0, 1000))

N = np.load(f'antiCVs/NBs.npy')
C = np.load(f'antiCVs/levels.npy')
nbs = np.arange(0.1, 5.1, 0.1)
target = 1e-11
tf = 10000

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "18"
    })
# fig, ax = plt.subplots(2,2)
# plt.subplots_adjust(wspace=0.01,hspace=0.01)
# positions = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])





for i, zeta in enumerate(zetas):
    # T = np.load(f'antiCVs/Topts{zeta}.npy')
    G = np.load(f'antiCVs/Gopts{zeta}.npy')
    anti = np.empty(np.shape(G))
    # pos = positions[i]
    for j, nbm in enumerate(nbs):


        def antiattarget(k):
            # this function will return the anti-correlation at the time the correlation variance is exiting threshold
            g = G[j][k]
            cl = levels[k]
            scale = 1 / (1 + nbm)
            corrmax = 2 * scale
            ic = np.array([-0.5 * zeta * scale * nbm, -scale * nbm, 0])
            t, state = utils.rel_simulation(zeta, g, 0, 0, ic, corrmax, target, tf)
            corr = utils.rel_corr_var(state) * (nbm + 1)
            anticor = (2 * state[0, :] + 1) * np.exp(2 * state[2, :]) * (nbm + 1)
            # Where are the indexes where we are under the target cv?
            entangledindex = np.nonzero(corr < cl)
            return anticor[entangledindex[0][-1]]


        with Pool(processes=15) as pool:
            corrsarray = pool.map(antiattarget, range(np.size(levels)))
        for n in range(np.size(levels)):
            anti[j][n] = corrsarray[n]

        print(f'nb = {nbm} done...')
    # CF = ax[pos[0]][pos[1]].contourf(C, N, np.log(anti), cmap='viridis', origin="lower")
    # ax[pos[0]][pos[1]].text(0.25, 200, f'$\zeta$ = {zeta}', bbox=dict(alpha=0.8, ec='k', fc='white', boxstyle='square'), size='16')
    # ax[pos[0]][pos[1]].set_xlabel(r"$\Delta^2_{t}$")
    # ax[pos[0]][pos[1]].set_ylabel(r"$n_m^b$")
    # ax[pos[0]][pos[1]].get_xaxis().set_visible(pos[0])
    # ax[pos[0]][pos[1]].get_yaxis().set_visible(not pos[1])
    print(f'zeta = {zeta} done...')
    np.save(f"antiCVs/anticorrs{zeta}", anti)
# bar = fig.colorbar(CF, ax=ax).set_label(label=r"$\log_{10} (2\bar n_{th} + 1)e^{2u}$", size=16)
# fig.savefig(f"antiCVs/anticor.pdf", format='pdf', dpi=1200)


