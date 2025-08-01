import numpy as np
from multiprocessing import Pool

"""
'optGSolver'
For computing gopt based on analytic expression. 
24 July 2025
Paul RB Hughes
"""

path = "pumpOptimizationData/paperData/analytic/"

C = np.load(path + "levels.npy")
N = np.load(path + "NBs.npy")
Gany = np.zeros(np.shape(N))
intercepts = np.load(path + "levels1.npy")
gints = np.load(path + "gopts1.npy")
zetas = [0.99, 0.992, 0.994, 0.996, 0.998]


for zeta in zetas:
    coeff = 3.4371619800745017 * (1 - zeta)**1.0025867074600905
    for i, nb in enumerate(N[:, 0]):


        def gfinderlevel(level):


            def function(k):  # k indexes the intercepts
                return coeff * nb / (gints[k] + 1) + intercepts[k] - level


            placeholder = np.abs([function(k) for k in range(np.size(gints))])
            optindex = np.argmin(placeholder)
            return gints[optindex]


        with Pool(processes=15) as pool:
            temp = pool.map(gfinderlevel, C[0, :])
        Gany[i, :] = temp
        print(f"nb {nb} done")
    np.save(path + f"Gopts{zeta}", Gany)
    print(f"zeta = {zeta} done")
