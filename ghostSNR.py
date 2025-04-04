import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": "14"
    })

"""
ghostSNR.py
Purpose: Ghost imaging SNR.
Date: 25 Mar 2025
Paul RB Hughes
"""

def dg2(n1th, n2th, u):
    return np.square((n1th + n2th + 1) * np.sinh(2*u)) * 0.25

def modepop(ntha, nthb, u):
    return ntha*np.square(np.cosh(u)) + (nthb + 1)*np.square(np.sinh(u))

epsilon = 0.2
n2th = np.arange(0.8, 6.01, 0.01)
n1th = epsilon #* np.ones(np.size(n2th))
u = np.arange(0.1, 1.51, 0.01)
U, N = np.meshgrid(u, n2th)
# U, N1 = np.meshgrid(u, n1th)
n1 = modepop(n1th, N, U)
n2 = modepop(N, n1th, U)

DeltaG2 = dg2(N, n1th, U)
SNR = DeltaG2/(n1 * n2)

fig, ax = plt.subplots()

# print(theta)

steadycor = ax.contourf(U, N, SNR, cmap='viridis', origin="lower")
# ploting isonums of n1
CS = ax.contour(U, N, n1, levels=[0., 1., 2., 4., 6., 8., 10., 12.])
ax.clabel(CS)

ax.set_xlabel("$u$")
ax.set_ylabel("$n_2^{th}$")
plt.tight_layout()

fig.colorbar(steadycor, ax=ax, label=r"$SNR$")

# plt.savefig("SNR.pdf", format='pdf', dpi=1200, bbox_inches='tight')
plt.show()
