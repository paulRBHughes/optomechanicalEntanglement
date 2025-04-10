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
Date: 7 Apr 2025
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

Nframe = 50
n1bs = np.flip(np.logspace(-7, -1, Nframe))
# n1bs = np.flip(np.linspace(0.25, 0.75, Nframe))
n2b = 1
epsilon = 0.01
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

fig, ax = plt.subplots()
levs = [0, 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 4]
# levs = np.arange(0, 6)
numLevels = [0.01, 0.03, 0.1, 0.3, 1.]
ax.set_xlabel("$\zeta$")
ax.set_ylabel("$g$")
cf = ax.contourf(Z, G, Gp, levs)
isonum = ax.contour(Z, G, Gp, numLevels)
mx, = ax.plot([], [], 'ro')
nb_text = ax.text(-0.85, 0.9, '')


def animate(i):
    n1b = n1bs[i]
    global cf, nb_text, isonum, levs, numLevels, mx
    SSvals = steady(Z, Gp, n1b, n2b)
    DeltaG2 = dg2(SSvals[1], SSvals[2], SSvals[0])
    SNR = DeltaG2/(modepop(SSvals[1], SSvals[2], SSvals[0]) * modepop(SSvals[2], SSvals[1], SSvals[0]))
    locs = np.unravel_index(np.nanargmax(SNR), SNR.shape)
    # print(np.nanmax(SNR))

    nb_text.set_text(f"$n_1^b = {n1b}$")

    for coll in cf.collections:
        coll.remove()

    for coll in isonum.collections:
        coll.remove()

    cf = ax.contourf(Z, G, SNR, levs, cmap='viridis', origin="lower")
    isonum = ax.contour(Z, G, modepop(SSvals[1], SSvals[2], SSvals[0]), numLevels)
    # mx = ax.plot(zs[locs[0]], gees[locs[1]], 'ro')
    mx.set_data(zs[locs[0]], gees[locs[1]])
    return mx,


fig.colorbar(cf, ax=ax, label=r"$SNR$")


anim = animation.FuncAnimation(fig, animate, frames=Nframe)
anim.save("ghost/what.gif", writer='pillow', fps=6)

fig.clf()

# ##
# zeta = 0.25
# # # n1b = 0.
# gmax = 1 - zeta**2 - epsilon
# gees = np.arange(0, gmax, 0.001)
# n2b = np.arange(0.01, 2.01, 0.01)
# N, Gn = np.meshgrid(n2b, gees)
# fig, ax = plt.subplots()
# ax.set_xlabel("$n_2^b$")
# ax.set_ylabel("$g$")
# nb_text = ax.text(0.5, 0.8, '')
# levs = np.append(levs, [10, 11, 12])
#
# cf = ax.contourf(N, Gn, Gn, levs, cmap='viridis', origin="lower")
# CS = ax.contour(N, Gn, Gn, levels=numLevels)
#
#
# def animate(i):
#     n1b = n1bs[i]
#     global cf, nb_text, CS, levs, numLevels, mx
#     SSvals = steady(zeta, Gn, n1b, N)
#     DeltaG2 = dg2(SSvals[1], SSvals[2], SSvals[0])
#     SNR = DeltaG2/(modepop(SSvals[1], SSvals[2], SSvals[0]) * modepop(SSvals[2], SSvals[1], SSvals[0]))
#     # locs = np.unravel_index(np.nanargmax(SNR), SNR.shape)
#
#     nb_text.set_text(f"$n_1^b = {n1b}$")
#
#     for coll in cf.collections:
#         coll.remove()
#
#     for coll in CS.collections:
#         coll.remove()
#
#     cf = ax.contourf(N, Gn, SNR, levs, cmap='viridis', origin="lower")
#     CS = ax.contour(N, Gn, modepop(SSvals[1], SSvals[2], SSvals[0]), numLevels)
#
#
#
# fig.colorbar(cf, ax=ax, label=r"$SNR$")
#
# anim = animation.FuncAnimation(fig, animate, frames=Nframe)
# anim.save("ghost/Nanimation2TENTHs.gif", writer='pillow', fps=6)
