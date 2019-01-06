import numpy as np
import numpy.random as npr
import numpy.linalg as npl

"""
fn 1: vector elementwise cube
"""

def fn1(x):
    return np.sum(np.power(x, 3))

def grad1(x):
    return 3. * np.power(x, 2)

def inv_grad1(z):
    """ functional inverse of the gradient """
    return np.sqrt(z / 3.)

def hess_mult1(x, v):
    """ hessian pre-multiplication as in Pearlmutter 1993 """
    return 6. * x * v

def hess_inv_mult1(z, v):
    """ inverse hessian pre-multiplication, the novel bit """
    return (0.5 * np.power(z / 3., -0.5)) * (v / 3.)

"""
fn 2: less easy one. assume cardinality x is 3
"""

def fn1(x):
    return x[0] * x[1] * x[2]

def grad1(x):
    return np.array([
        x[1] * x[2],
        x[0] * x[2],
        x[0] * x[1],
    ])

def inv_grad(z):
    return np.array([
        np.sqrt((x[1] * x[2]) / x[0}),
        np.sqrt((x[0] * x[2]) / x[1}),
        np.sqrt((x[0] * x[1]) / x[2}),
    ])

def hess_mult(x, v):
    return np.array([
        v[1] * x[2] + v[2] * x[1],
        v[0] * x[2] + v[2] * x[0],
        v[0] * x[1] + v[1] * x[0],
        ])

def hess_inv_mult(z, v):
    return np.array([
        np.power((z[1] * z[2]) / z[0], -0.5) * (((v[1] * z[2] + v[2] * z[1]) * z[0] - (v[0] * z[1] * z[2])) / (z[0] ** 2)),
        np.power((z[0] * z[2]) / z[1], -0.5) * (((v[0] * z[2] + v[2] * z[0]) * z[1] - (v[1] * z[0] * z[2])) / (z[1] ** 2)),
        np.power((z[0] * z[1]) / z[2], -0.5) * (((v[0] * z[1] + v[1] * z[0]) * z[2] - (v[2] * z[0] * z[1])) / (z[2] ** 2)),
        ])

if __name__ == "__main__":
    pass
