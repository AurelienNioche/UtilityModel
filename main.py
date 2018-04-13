import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize


def u_monkey(m, w, max_m=10):

    return (m/max_m) ** (1-w)


def u_CRRA(m, w):

    if m == 0:
        return None

    if w == 1:
        return np.log(m)

    return m ** (1-w) / (1-w)


def u_CARA(m, w):

    if w == 0:
        return m

    return (1 - np.exp(-w*m)) / w


def omega_CRRA(p1, m1, p2, m2):
    return (np.log((p1*m1) / p2) - np.log(m2)) / (np.log(m1) - np.log(m2))


def omega_CARA(p1, m1, p2, m2):

    def f(w):
        return np.absolute(p1*u_CARA(m1, w) - p2*u_CARA(m2, w))

    # res = scipy.optimize.minimize(f, x0=(1, ), bounds=((-2, 2), ))
    # return res.x[0]
    res = scipy.optimize.fmin(f, x0=1)
    return res[0]


def r(omega, lbd, w):
    return np.exp(omega * lbd) / (np.exp(omega * lbd) + np.exp(lbd * w))


def rho(lbd, w, p1, m1, p2, m2, omega_func):

    omega = omega_func(p1, m1, p2, m2)
    return r(omega=omega, lbd=lbd, w=w)


def softmax(lbd, w, p1, m1, p2, m2, u):

    U1 = p1 * u(m1, w)
    U2 = p2 * u(m2, w)

    v1 = np.exp(lbd * U1)
    v2 = np.exp(lbd * U2)

    return v1 / (v1 + v2)


def main():

    p1, m1 = 0.5, 0.5  # Riskiest
    p2, m2 = 1, 0.25  # The more safe

    lbd = 7

    x = np.linspace(-2, 2, 100)

    y0 = np.zeros(len(x))
    y1 = np.zeros(len(x))
    y2 = np.zeros(len(x))
    y3 = np.zeros(len(x))
    y4 = np.zeros(len(x))

    for i, w in enumerate(x):
        y0[i] = softmax(lbd=lbd, w=w, p1=p1, p2=p2, m1=m1, m2=m2, u=u_CRRA)
        y1[i] = softmax(lbd=lbd, w=w, p1=p1, p2=p2, m1=m1, m2=m2, u=u_monkey)
        y2[i] = softmax(lbd=lbd, w=w, p1=p1, p2=p2, m1=m1, m2=m2, u=u_CARA)
        y3[i] = rho(lbd=lbd, w=w, p1=p1, p2=p2, m1=m1, m2=m2, omega_func=omega_CRRA)
        y4[i] = rho(lbd=lbd, w=w, p1=p1, p2=p2, m1=m1, m2=m2, omega_func=omega_CARA)

    plt.plot(x, y0, label="Softmax / CRRA")
    plt.plot(x, y1, label="Softmax / monkey")
    plt.plot(x, y2, label="Softmax / CARA")
    plt.plot(x, y3, label="RPM / CRRA - monkey")
    plt.plot(x, y4, label="RPM / CARA")
    plt.legend()
    plt.show()

    x = np.linspace(0, 10, 100)
    y0 = np.zeros(len(x))
    y1 = np.zeros(len(x))
    y2 = np.zeros(len(x))

    # ----------------------------- #

    w = 0.5
    for i, m in enumerate(x):
        y0[i] = u_CRRA(m, w)
        y1[i] = u_CARA(m, w)
        y2[i] = u_monkey(m, w, max_m=max(x))

    plt.plot(x, y0, label="CRRA")
    plt.plot(x, y1, label="CARA")
    plt.plot(x, y2, label="Monkey")
    plt.legend()
    plt.show()

    # ------------------------------ #

    p1, m1 = 0.5, 2
    p2, m2 = 1, 1

    lbd1 = 1
    lbd2 = 100

    x = np.linspace(-2, 2, 100)

    y0 = np.zeros(len(x))
    y1 = np.zeros(len(x))
    y2 = np.zeros(len(x))
    y3 = np.zeros(len(x))

    for i, w in enumerate(x):
        y0[i] = softmax(lbd=lbd1, w=w, p1=p1, p2=p2, m1=m1, m2=m2, u=u_monkey)
        y1[i] = rho(lbd=lbd1, w=w, p1=p1, p2=p2, m1=m1, m2=m2, omega_func=omega_CRRA)
        y2[i] = softmax(lbd=lbd2, w=w, p1=p1, p2=p2, m1=m1, m2=m2, u=u_monkey)
        y3[i] = rho(lbd=lbd2, w=w, p1=p1, p2=p2, m1=m1, m2=m2, omega_func=omega_CRRA)

    plt.plot(x, y0, label="Softmax with monkey; $\lambda$ = {}".format(lbd1), c="C0", linestyle="-")
    plt.plot(x, y1, label="RPM with monkey; $\lambda$ = {}".format(lbd1), c="C1", linestyle="-")
    plt.plot(x, y2, label="Softmax with monkey; $\lambda$ = {}".format(lbd2), c="C0", linestyle="--")
    plt.plot(x, y3, label="RPM with monkey; $\lambda$ = {}".format(lbd2), c="C1", linestyle="--")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    main()
