import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import lineStyles

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
    n1th = ((1 + zeta) * np.square(np.cosh(u)) * n1b + (1 - zeta) * np.square(np.sinh(u)) * (1 + n2b)) / (
                1 + zeta * np.cosh(2 * u))
    n2th = ((1 - zeta) * np.square(np.cosh(u)) * n2b + (1 + zeta) * np.square(np.sinh(u)) * (1 + n1b)) / (
                1 - zeta * np.cosh(2 * u))
    return np.array([u, n1th, n2th])


def dg2(n1th, n2th, u):
    return np.square(np.sinh(2*u)) * 0.25 * (2 * n1th * n2th + n1th + n2th + 1)


def modepop(ntha, nthb, u):
    return ntha*np.square(np.cosh(u)) + (nthb + 1)*np.square(np.sinh(u))


def g2(n1th, n2th, u):
    return dg2(n1th, n2th, u) + modepop(n1th, n2th, u) * modepop(n2th, n1th, u)


def placeholder(nath, nbth, u):
    c = np.cosh(u)
    s = np.sinh(u)
    return (nath**2 - nbth) * c**4 + 4 * nath * (nbth + 1) * c**2 * s**2 + (nbth + 1) * (nbth + 2) * s**4 + nath * c**2 + (nbth + 1) * s**2


def backgroundnoise(n1th, n2th, u):
    cs = np.cosh(u) * np.sinh(u)
    n1 = modepop(n1th, n2th, u)
    n2 = modepop(n2th, n1th, u)
    a = n1th * n2th * cs**2 + (n1th + 1) * (n2th + 1) * cs**2
    return (n1**2 + n2**2 + a) * a


def noisefunction(n1th, n2th, u):
    c = np.cosh(u)
    s = np.sinh(u)
    n1 = modepop(n1th, n2th, u)
    n2 = modepop(n2th, n1th, u)
    a = placeholder(n1th, n2th, u) * placeholder(n2th, n1th, u)
    b = (c**2 * s**2 * n1th * n2th * (2 * c**2 * (n1th - 1) + 2 * s**2 * (n2th + 1) + 1)
         * (2 * c**2 * (n2th - 1) + 2 * s**2 * (n1th + 1) + 1))
    c = (c**2 * s**2 * (n1th + 1) * (n2th + 1) * (2 * c**2 * n1th + 2 * s**2 * (n2th + 2) + 1)
         * (2 * c**2 * n2th + 2 * s**2 * (n1th + 2) + 1))
    d = n1th * n2th * (n1th - 1) * (n2th - 1) * c**4 * s**4
    e = (n1th + 1) * (n1th + 2) * (n2th + 1) * (n2th + 2) * c**4 * s**4
    signoise = a + b + c + d + e - g2(n1th, n2th, u) **2
    bgnoise = backgroundnoise(n1th, n2th, u)
    return np.sqrt(bgnoise)



fig, ax = plt.subplots()

n1b = 1e-7
n2b = 0.1
epsilon = 1e-2
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
noise = noisefunction(SSvals[1], SSvals[2], SSvals[0])
SNR = DeltaG2 / noise

levs = np.linspace(-15, 3, 13)
numLevels = [0.01, 0.03, 0.1, 0.3, 1.]



steadycor = ax.contourf(Z, G, np.log10(SNR), cmap='viridis', origin="lower")
# ploting isonums of n1
CS = ax.contour(Z, G, modepop(SSvals[1], SSvals[2], SSvals[0]), levels=numLevels, linestyles="dotted")
# ax.clabel(CS)
C2S = ax.contour(Z, G, modepop(SSvals[2], SSvals[1], SSvals[0]), levels=numLevels, linestyles="dashed")
ax.clabel(C2S)

locs = np.unravel_index(np.nanargmax(SNR), SNR.shape)
ax.plot(zs[locs[0]], gees[locs[1]], 'ro')

ax.set_xlabel("$\zeta$")
ax.set_ylabel("$g$")
plt.tight_layout()

fig.colorbar(steadycor, ax=ax, label=r"$SNR (dB)$")

# plt.savefig("ghost/VRefined.pdf", format='pdf', dpi=1200, bbox_inches='tight')

print(np.nanmax(10*np.log10(SNR)))
print(np.nanmin(10*np.log10(SNR)))

##
# zeta = 0.25
# # n1b = 0.
# gmax = 1 - zeta**2 - epsilon
# gees = np.arange(0, gmax, 0.001)
# n2b = np.arange(0.25, 2.01, 0.01)
# N, Gn = np.meshgrid(n2b, gees)
#
# SSvals = steady(zeta, Gn, n1b, N)
#
# DeltaG2 = dg2(SSvals[1], SSvals[2], SSvals[0])
# V = DeltaG2/(DeltaG2 + 2 * modepop(SSvals[1], SSvals[2], SSvals[0]) * modepop(SSvals[2], SSvals[1], SSvals[0]))
#
# fig, ax = plt.subplots()
#
# steadycor = ax.contourf(N, Gn, V, levs, cmap='viridis', origin="lower")
# # ploting isonums of n1
# CS = ax.contour(N, Gn, modepop(SSvals[1], SSvals[2], SSvals[0]), levels=numLevels)
# ax.clabel(CS)
#
# ax.set_xlabel("$n_2^b$")
# ax.set_ylabel("$g$")
# plt.tight_layout()
#
# fig.colorbar(steadycor, ax=ax, label=r"$V$")
#
#
# # plt.savefig("ghost/numberSNR.pdf", format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
