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
fig, ax = plt.subplots()
figi, axi = plt.subplots()
figz, axz = plt.subplots()

for i, zeta in enumerate(zetas):
    if not loadData:
        gees = np.array([])
        slopes = np.array([])
        ints = np.array([])
        G = np.load(path + f"Gopts{zeta}.npy")
        for g in G.flatten():
            nbs = N[G == g]  # where is gopt the same
            nbs, inds = np.unique(nbs, True)
            if np.size(nbs) < 2:
                continue
            levs = C[G == g]
            levs = levs[inds]
            regression = linregress(levs, nbs)  # note that this regression is same as plot, but opposite the eqn
            gees = np.append(gees, g)
            slopes = np.append(slopes, regression.slope)
            ints = np.append(ints, regression.intercept)
        np.save(path+f"gees{zeta}", gees)
        np.save(path+f"slopes{zeta}", slopes)
        np.save(path+f"ints{zeta}", ints)
    else:
        gees = np.load(path+f"gees{zeta}.npy")
        slopes = np.load(path + f"slopes{zeta}.npy")
        ints = np.load(path + f"ints{zeta}.npy")
    ax.scatter(gees, slopes, label=zeta, c=colors[i], s=2)
    axi.scatter(gees, -ints/slopes, label=zeta, c=colors[i], s=2)  # contorted to give beta(gopt) which appears independent of zeta
    regre = linregress(gees, slopes)  # as it happens, this is basically linear with slopes determined by zeta
    slopeOfSlopes[i] = regre.slope

regagain = linregress(np.log(1-zetas), np.log(slopeOfSlopes))  # loglog regression
k = regagain.slope
print(k)  # this is the power of zeta (negative in eqn)
print(np.exp(-regagain.intercept))  # this will give the rest of the slope

ax.set_ylim(0, 10000)
ax.set_xlabel("$g_{opt}$")
ax.set_ylabel("slope")
ax.legend(title=r"$\zeta$", loc="upper left", markerscale=4)

axi.set_xlabel("$g_{opt}$")
axi.set_ylabel("intercept")

axz.scatter(1 - zetas, slopeOfSlopes, c=colors[-1])
axz.plot(1-zetas, np.exp(regagain.intercept) * (1-zetas)**k)  # check how close this is

axz.set_xlabel("$1 - \zeta$")
axz.set_ylabel(r"slope of slope")
axz.set_xscale('log')
axz.set_yscale('log')
if save:
    fig.savefig(path+"slopes.jpg", format='jpg', dpi=600, bbox_inches='tight')
    figi.savefig(path+"ints.jpg", format='jpg', dpi=600, bbox_inches='tight')
    figz.savefig(path + "zetadep.jpg", format='jpg', dpi=600, bbox_inches='tight')
plt.show()
