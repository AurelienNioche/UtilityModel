from scipy import optimize


def f(x, a, b):

    return x**2  # + a + b


res = optimize.fmin(f, x0 =(2, ), args=(3, 4))
print(res[0])