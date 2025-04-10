import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })

"""
ghostSNR.py
Purpose: Ghost imaging SNR for squeezing produced by an OPO or whatever in the steady state?
Date: 27 Mar 2025
Paul RB Hughes
"""

def steady(zeta, g, n1b, n2b):
    # if g == 0:
    #     return np.zeros([3, np.size(g)])
    u = 0.5 * np.arctanh(g)
    n1th = ((1 - zeta) * np.square(np.cosh(u)) * n1b + (1 - zeta) * np.square(np.sinh(u)) * (1 + n2b)) / (
                1 + zeta * np.cosh(2 * u))
    n2th = ((1 + zeta) * np.square(np.cosh(u)) * n2b + (1 + zeta) * np.square(np.sinh(u)) * (1 + n1b)) / (
                1 - zeta * np.cosh(2 * u))
    return np.array([u, n1th, n2th])


def dg2(n1th, n2th, u):
    return np.square((n1th + n2th + 1) * np.sinh(2*u)) * 0.25


def modepop(ntha, nthb, u):
    return ntha*np.square(np.cosh(u)) + (nthb + 1)*np.square(np.sinh(u))


fig, ax = plt.subplots()

n1b = 0.001
n2b = 0.5
epsilon = 0.01
# zmax = 0.6
# gmax = 1 - zmax**2
zs = np.arange(-1, 1, 0.001)
gees = np.arange(0, 1, 0.001)
G, Z = np.meshgrid(gees, zs)
Gp = np.copy(G)
for i, gs in enumerate(G):
    z = Z[i][0]
    cond = 1 - z**2
    for j, g in enumerate(gs):
        if g**2 >= cond - epsilon:
            Gp[i][j:] = np.NaN
            break

SSvals = steady(Z, Gp, n1b, n2b)

DeltaG2 = dg2(SSvals[1], SSvals[2], SSvals[0])
SNR = DeltaG2/(modepop(SSvals[1], SSvals[2], SSvals[0]) * modepop(SSvals[2], SSvals[1], SSvals[0]))

levs = [0, 0.5, 1, 1.1, 1.2, 1.3, 1.4]
numLevels = [0.01, 0.03, 0.1, 0.3, 1.]

steadycor = ax.contourf(Z, G, (SNR), np.append(levs, [1.5, 1.6]), cmap='viridis', origin="lower")
# ploting isonums of n1
CS = ax.contour(Z, G, modepop(SSvals[1], SSvals[2], SSvals[0]), levels=numLevels)
ax.clabel(CS)

ax.set_xlabel("$\zeta$")
ax.set_ylabel("$g$")
plt.tight_layout()

fig.colorbar(steadycor, ax=ax, label=r"$SNR$")

plt.savefig("ghost/zetaSNR.pdf", format='pdf', dpi=1200, bbox_inches='tight')

##
zeta = 0.25
# n1b = 0.
gmax = 1 - zeta**2 - epsilon
gees = np.arange(0, gmax, 0.001)
n2b = np.arange(0.25, 2.01, 0.01)
N, Gn = np.meshgrid(n2b, gees)

SSvals = steady(zeta, Gn, n1b, N)

DeltaG2 = dg2(SSvals[1], SSvals[2], SSvals[0])
SNR = DeltaG2/(modepop(SSvals[1], SSvals[2], SSvals[0]) * modepop(SSvals[2], SSvals[1], SSvals[0]))

fig, ax = plt.subplots()

steadycor = ax.contourf(N, Gn, SNR, np.append(levs, [1.5, 1.6, 1.7, 1.8, 1.9]), cmap='viridis', origin="lower")
# ploting isonums of n1
CS = ax.contour(N, Gn, modepop(SSvals[1], SSvals[2], SSvals[0]), levels=numLevels)
ax.clabel(CS)

ax.set_xlabel("$n_2^b$")
ax.set_ylabel("$g$")
plt.tight_layout()

fig.colorbar(steadycor, ax=ax, label=r"$SNR$")


plt.savefig("ghost/numberSNR.pdf", format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
