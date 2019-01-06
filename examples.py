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
fn 2: less easy one
"""

def fn1(x):
    return x[0] * x[1] * x[2]

if __name__ == "__main__":
    pass
