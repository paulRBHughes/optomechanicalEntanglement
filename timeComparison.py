import utils
import numpy as np
from multiprocessing import Pool

"""
timeComparison.py
Purpose: to consider the analytic pump strengths found in optGSolver.py and compare the time below the level with the numerical solution
29 July 2025
Paul RB Hughes
"""

zetas = [0.99, 0.992, 0.994, 0.996, 0.998]
path = "pumpOptimizationData/paperData/analytic/"
C = np.load(path + "levels.npy")
N = np.load(path + "NBs.npy")
target = 1e-11
tf = 30

for zeta in zetas:
    tauanyl = np.zeros(np.shape(N))
    ganyl = np.load(path + f"Gopts{zeta}.npy")
    for i, nb in enumerate(N[:, 0]):
        scale = 1/(1 + nb)
        corrmax = scale


        def timeunder(j):
            level = C[0, j]
            ic = np.array([-0.5 * zeta * scale * nb, -scale * nb, 0])  # assumes large bath for mech
            t, state = utils.rel_simulation(zeta, ganyl[i, j], 0, 0, ic, corrmax, target, tf)
            corr = utils.rel_corr_var(state)
            mc = np.min(corr)
            # now I want to find time it is entangled
            if mc > level * scale:
                return 0
            entangledindex = np.nonzero(corr < level * scale)
            return t[entangledindex[0][-1]] - t[entangledindex[0][0]]


        with Pool(processes=15) as pool:  # multiprocess parallelism, makes it go way faster
            taus = pool.map(timeunder, range(np.size(C[0, :])))
        tauanyl[i, :] = taus

        print(f"nb = {nb} done...")
    np.save(path + f"taus{zeta}", tauanyl)
    print(f"zeta = {zeta} done...")
