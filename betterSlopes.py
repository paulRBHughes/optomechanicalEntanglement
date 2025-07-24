import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

import utils
from multiprocessing import Pool

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "16"
    })

"""
'betterSlopes.py'
To use the data that made fig 12 to get the slopes
21 July 2025
Paul RB Hughes
"""

path = 'pumpOptimizationData/paperData/'
save = False
loadData = True

N = np.load(path+f'NBs.npy')
C = np.load(path+f'levels.npy')

colors = ["lightseagreen", "mediumspringgreen", "mediumseagreen", "seagreen", "darkgreen"]

zetas = np.array([0.99, 0.992, 0.994, 0.996, 0.998])
slopeOfSlopes = np.empty(np.size(zetas))
intsOfSlopes = np.empty(np.size(zetas))
# fig, ax = plt.subplots()
figi, axi = plt.subplots()

# figz, axz = plt.subplots()
for i, zeta in enumerate(zetas):
    if not loadData:
        gees = np.array([])
        slopes = np.array([])
        ints = np.array([])
        G = np.load(path + f"Gopts{zeta}.npy")
        for g in G.flatten():
            nbs = N[G == g]
            nbs, inds = np.unique(nbs, True)
            if np.size(nbs) < 2:
                continue
            levs = C[G == g]
            levs = levs[inds]
            regression = linregress(levs, nbs)
            gees = np.append(gees, g)
            slopes = np.append(slopes, regression.slope)
            ints = np.append(ints, regression.intercept)
            # print(f"g = {g} done")
        np.save(path+f"gees{zeta}", gees)
        np.save(path+f"slopes{zeta}", slopes)
        np.save(path+f"ints{zeta}", ints)
    else:
        gees = np.load(path+f"gees{zeta}.npy")
        slopes = np.load(path + f"slopes{zeta}.npy")
        ints = np.load(path + f"ints{zeta}.npy")
    # ax.scatter(gees - 1, slopes, label=zeta, c=colors[i], s=2)
    axi.scatter(gees - 1, -ints/slopes, label=zeta, c=colors[i], s=2)
    # regre = linregress(gees - 1, slopes)
    # slopeOfSlopes[i] = regre.slope
# ax.set_ylim(0, 10000)
# ax.set_xlabel("$g_{opt} - 1$")
# ax.set_ylabel("slope")
# ax.legend(title=r"$\zeta$", loc="upper left", markerscale=4)

# axi.set_ylim(0, 1)
# x = np.arange(1, 120, 0.1)
# axi.plot(x, 1/np.cosh(x) * 2, color = "red")
# axi.set_xscale('log')
# axi.set_yscale('log')
axi.set_xlabel("$g_{opt} - 1$")
axi.set_ylabel("intercept")
axi.legend(title=r"$\zeta$", loc="upper right", markerscale=4)
if save:
    # fig.savefig(path+"slopes.jpg", format='jpg', dpi=600, bbox_inches='tight')
    figi.savefig(path+"ints.jpg", format='jpg', dpi=600, bbox_inches='tight')
plt.show()
#
# regagain = linregress(np.log(1-zetas), np.log(slopeOfSlopes))
# k = regagain.slope
# print(k)
# print(np.exp(regagain.intercept))
#
# axz.scatter(1-zetas, slopeOfSlopes, c=colors[-1])
# axz.plot(1-zetas, np.exp(regagain.intercept) * (1-zetas)**k)
# axz.set_xlabel("$1 - \zeta$")
# axz.set_ylabel(r"slope of slope")
# axz.set_xscale('log')
# axz.set_yscale('log')
# if save:
#     figz.savefig(path+"slopeOfSlopes.pdf", format='pdf', dpi=1200, bbox_inches='tight')
#
# figi, axi = plt.subplots()
# regint = linregress(np.log(1-zetas), np.log(intsOfSlopes))
# k = regint.slope
# print(k)
# print(np.exp(regint.intercept))
# axi.plot(1-zetas, np.exp(regint.intercept) * (1-zetas)**k)
# axi.scatter(1 - zetas, intsOfSlopes, c=colors[-1])
# axi.set_xlabel("$1 - \zeta$")
# axi.set_ylabel(r"intercept of slope")
# axi.set_xscale('log')
# axi.set_yscale('log')
# plt.show()
