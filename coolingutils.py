"""
Project: optomechanical cooling
Purpose: Model ODE equations and make plots to cool mechanical mode
July 2024
Paul RB Hughes
"""

# Import
import numpy as np
from numba import njit

"""
Need to use the full equation set (for now, since gamma is static - interaction)
"""


# @njit
def eom(zeta, gamma, nb1, nb2, state):
    n1 = state[0]
    n2 = state[1]
    theta = state[2]
    deltan = n2 - n1
    deltanb = nb2 - nb1
    twonth = n1 + n2
    twonb = nb1 + nb2
    dn1 = ((1 + zeta) * nb1 * np.square(np.cos(theta)) + (1 - zeta) * nb2 * np.square(np.sin(theta)) -
           (1 + zeta * np.cos(2 * theta)) * n1)
    dn2 = ((1 - zeta) * nb2 * np.square(np.cos(theta)) + (1 + zeta) * nb1 * np.square(np.sin(theta)) -
           (1 - zeta * np.cos(2 * theta)) * n2)
    dtheta = -gamma * 0.5 + (zeta * (twonb - twonth) - deltanb) * np.sin(2 * theta)/deltan
    return np.array([dn1, dn2, dtheta])


def rk4slope(zeta, gamma, nb1, nb2, state, dt):
    k1 = eom(zeta, gamma, nb1, nb2, state)
    k2 = eom(zeta, gamma, nb1, nb2, state + 0.5 * k1 * dt)
    k3 = eom(zeta, gamma, nb1, nb2, state + 0.5 * k2 * dt)
    k4 = eom(zeta, gamma, nb1, nb2, state + k3 * dt)
    return k1/6 + k2/3 + k3/3 + k4/6


def simulation(zeta, gamma, nb1, nb2, initial, target, tf):
    # target will be the target precision of a thermal population
    dt = 0.01  # arb
    t = np.zeros(100000000)  # arb
    state = np.zeros([3, 100000000])
    state[:, 0] = initial[:]
    i = 0
    while t[i] < tf:  # standard dynamic timestep trick
        test = 30 * dt * target
        # two steps of dt
        k = rk4slope(zeta, gamma, nb1, nb2, state[:, i], dt)
        check1a = state[:, i] + k * dt
        k = rk4slope(zeta, gamma, nb1, nb2, check1a, dt)
        check1 = check1a + k * dt
        # one step of 2dt
        k = rk4slope(zeta, gamma, nb1, nb2, state[:, i], 2*dt)
        check2 = state[:, i] + 2 * k * dt
        # checking error estimate
        check = np.max(np.absolute(check1[:2] - check2[:2]))  # only care about populations
        if check < 0.5 * test:
            rho = 2
        else:
            rho = test/check  # taking faith in old code
        if rho > 1:
            t[i + 1] = t[i] + dt
            state[:, i + 1] = check1a
            # state[2:, i + 1] = state[2:, i + 1] % 2*np.pi - np.pi
            i += 1
        dt = dt * rho**0.25

    t = t[:i-1]
    state = state[:, :i - 1]
    return t, state


@njit
def bath(omega, T):
    C = 75.3862e-12
    x = np.exp(C*omega/T)
    return 1/(x - 1)

