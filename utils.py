"""
Project: 2 Mode Thermal Squeezing in Thermal Bath
Purpose: Model ODE equations and make plots to see how things like seperability are effected by thermal baths
May 2024
Paul RB Hughes
"""

# Import
import numpy as np
from numba import njit
from functools import partial

#@njit
def thermal_eom(zeta, mode_bath, other_bath, nth, u):
    # Function to calculate the time derivative of the (thermal) population nth
    # zeta is the ratio of gamma- to gamma+, other variables define the state
    # BECAUSE gamma- is not symmetric, when calculating n2 use -zeta!!!!!!!!!!!!!!!!
    dn = (1+zeta) * 0.5 * (1 + np.cosh(2*u)) * (mode_bath - nth) + (1-zeta) * 0.5 * (np.cosh(2*u) - 1) * (other_bath + nth + 1)
    return dn


def eom(zeta, g, B, batha, bathb, state):
    # caluculate the time derivative of the squeeze parameter amplitude
    # a must correspond to the mode that has positive gamma-, i.e. n1 unless I change something
    omegasq = 1/(16*np.square(B) + 1)
    q = omegasq * np.square(g)/(1 + zeta)
    p = np.square(g) * np.square(B)/(2 * (1 + zeta)*(np.square(B) + 16))
    r = np.square(g)/(2 * (1 + zeta))
    w = np.square(g) / (1 + zeta) * (np.square(B) + 8)/(np.square(B) + 16)
    na = state[0]
    nb = state[1]
    u = state[2]
    dn1 = thermal_eom(zeta, batha, bathb, na, u) + (q * na + p)* np.square(np.sinh(u))
    dn2 = thermal_eom(-zeta, bathb, batha, nb, u) + (-q * na + r)* np.square(np.cosh(u))
    du = 0.5*(g - np.sinh(2*u) * (batha + bathb + 1 + zeta*(batha - na + nb - bathb) + w + q * (nb - na))/(na + nb + 1))
    return np.array([dn1, dn2, du])


def rel_eom(zeta, g, av_bath, diff_bath, state):
    # same as above but uses the average and difference numbers
    #state = [naverage, ndifference, u]
    av = state[0]
    diff = state[1]
    u = state[2]
    dav = 0.5 * ( ((2*av_bath + 1) + zeta*(diff - diff_bath))*np.cosh(2*u) - 2*av - 1)
    ddiff = diff_bath - diff - zeta*(2*av_bath + 1 - np.cosh(2*u) * (2*av + 1))
    du = 0.5 * (g - np.sinh(2*u)*(2*av_bath + 1 + zeta*(diff - diff_bath))/(2*av + 1))
    return np.array([dav, ddiff, du])


def rk4slope(zeta, g, B, batha, bathb, state, dt):
    k1 = eom(zeta, g, B, batha, bathb, state)
    k2 = eom(zeta, g, B, batha, bathb, state + 0.5 * k1 * dt)
    k3 = eom(zeta, g, B, batha, bathb, state + 0.5 * k2 * dt)
    k4 = eom(zeta, g, B, batha, bathb, state + k3 * dt)
    return k1/6 + k2/3 + k3/3 + k4/6


def pump_rk4slope(zeta, g, avb, db, state, t, dt):
    k1 = rel_eom(zeta, g(t), avb, db, state)
    k2 = rel_eom(zeta, g(t + 0.5 * dt), avb, db, state + 0.5 * k1 * dt)
    k3 = rel_eom(zeta, g(t + 0.5 * dt), avb, db, state + 0.5 * k2 * dt)
    k4 = rel_eom(zeta, g(t + dt), avb, db, state + k3 * dt)
    return k1/6 + k2/3 + k3/3 + k4/6


def pump_simulation(zeta, g_func, avb, db, initial, target, tf, *args):
    # same as above w
    g = partial(g_func, *args)
    dt = 0.01  # arb
    t = np.zeros(100000000)  # arb
    state = np.zeros([3, 100000000])
    state[:, 0] = initial[:]
    i = 0
    while t[i] < tf:  # standard dynamic timestep trick
        test = 30 * dt * target
        # two steps of dt
        k = pump_rk4slope(zeta, g, avb, db, state[:, i], t[i], dt)
        check1a = state[:, i] + k * dt
        k = pump_rk4slope(zeta, g, avb, db, check1a, t[i] + dt, dt)
        check1 = check1a + k * dt
        # one step of 2dt
        k = pump_rk4slope(zeta, g, avb, db, state[:, i], t[i], 2*dt)
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
            i += 1
        dt = dt * rho**0.25

    t = t[:i-1]
    state = state[:, :i - 1]
    return t, state


def simulation(zeta, g, B, bath1, bath2, initial, target, tf):
    # target will be the target precision of a thermal population
    dt = 0.01  # arb
    t = np.zeros(100000000)  # arb EXCEPT WHEN I RUN OUT OF MEMORY WHAT
    state = np.zeros([3, 100000000])
    state[:, 0] = initial[:]
    i = 0
    while t[i] < tf:  # standard dynamic timestep trick
        test = 30 * dt * target
        # two steps of dt
        k = rk4slope(zeta, g, B, bath1, bath2, state[:, i], dt)
        check1a = state[:, i] + k * dt
        k = rk4slope(zeta, g, B, bath1, bath2, check1a, dt)
        check1 = check1a + k * dt
        # one step of 2dt
        k = rk4slope(zeta, g, B, bath1, bath2, state[:, i], 2*dt)
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
            i += 1
        dt = dt * rho**0.25

    t = t[:i-1]
    state = state[:, :i - 1]
    return t, state


def rel_rk4slope(zeta, g, avb, db, state, dt):
    k1 = rel_eom(zeta, g, avb, db, state)
    k2 = rel_eom(zeta, g, avb, db, state + 0.5 * k1 * dt)
    k3 = rel_eom(zeta, g, avb, db, state + 0.5 * k2 * dt)
    k4 = rel_eom(zeta, g, avb, db, state + k3 * dt)
    return k1/6 + k2/3 + k3/3 + k4/6


def rel_simulation(zeta, g, avb, db, initial, target, tf):
    # same as above w
    dt = 0.01  # arb
    t = np.zeros(100000000)  # arb
    state = np.zeros([3, 100000000])
    state[:, 0] = initial[:]
    i = 0
    while t[i] < tf:  # standard dynamic timestep trick
        test = 30 * dt * target
        # two steps of dt
        k = rel_rk4slope(zeta, g, avb, db, state[:, i], dt)
        check1a = state[:, i] + k * dt
        k = rel_rk4slope(zeta, g, avb, db, check1a, dt)
        check1 = check1a + k * dt
        # one step of 2dt
        k = rel_rk4slope(zeta, g, avb, db, state[:, i], 2*dt)
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
            i += 1
        dt = dt * rho**0.25

    t = t[:i-1]
    state = state[:, :i - 1]
    return t, state


def correlation_var(state):
    return (1 + state[0, :] + state[1, :]) * np.exp(-2 * state[2, :])


def rel_corr_var(state):
    # same as above but using other n
    return (2*state[0, :] + 1)*np.exp(-2*state[2, :])


def rel_anti(state):
    # same as above but using other n
    return (2*state[0, :] + 1)*np.exp(2*state[2, :])

