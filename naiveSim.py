import numpy as np
import matplotlib.pyplot as plt
import utils

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": "28"
    })

zeta = 0.5
g0 = 0.5
n1b = 0.2
n2b = 0.2
ic = np.array([n1b, n2b, 0])
target = 0.01
tf = 20

fig, ax = plt.subplots(2)
t, state = utils.simulation(zeta, g0, n1b, n2b, ic, target, tf)
corr = utils.correlation_var(state)
ax[0].plot(t, corr)
ax[1].plot(t, state[1] - state[0])
plt.show()