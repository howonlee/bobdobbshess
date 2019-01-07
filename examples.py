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

def hess_inv_mult1(x, v):
    """ inverse hessian pre-multiplication, the novel bit. I switched vars"""
    return (0.5) * (v / (3. * x))

def test_f1():
    xs = npr.randn(3)
    vs = npr.randn(3)
    print("xs: ", xs)
    print("vs: ", vs)
    print("fn 1: ", fn1(xs))
    print("gradient of fn 1: ", grad1(xs))
    print("inv grad of grad of fn 1 should just be abs xs: ", inv_grad1(grad1(xs)))
    print("hess mult of vs: ", hess_mult1(xs, vs))
    print("inv hess mult of hess mult of vs should just be vs: ",
            hess_inv_mult1(xs, hess_mult1(xs, vs)))

"""
fn 2: less easy one. assume cardinality x is 3
"""

def fn2(x):
    return x[0] * x[1] * x[2]

def grad2(x):
    return np.array([
        x[1] * x[2],
        x[0] * x[2],
        x[0] * x[1],
    ])

def inv_grad2(z):
    return np.array([
        np.sqrt((z[1] * z[2]) / z[0]),
        np.sqrt((z[0] * z[2]) / z[1]),
        np.sqrt((z[0] * z[1]) / z[2]),
    ])

def hess_mult2(x, v):
    return np.array([
        v[1] * x[2] + v[2] * x[1],
        v[0] * x[2] + v[2] * x[0],
        v[0] * x[1] + v[1] * x[0],
        ])

def hess_inv_mult2(x, v):
    denom = 2. * x[0] * x[1] * x[2]
    return np.array([
        ((v[1] * x[0] * x[1]) + (v[2] * x[0] * x[2]) - (v[0] * x[0] * x[0])) / denom,
        ((v[0] * x[1] * x[0]) + (v[2] * x[1] * x[2]) - (v[1] * x[1] * x[1])) / denom,
        ((v[0] * x[0] * x[2]) + (v[1] * x[1] * x[2]) - (v[2] * x[2] * x[2])) / denom,
        ])

def test_f2():
    xs = npr.randn(3)
    vs = npr.randn(3)
    print("xs: ", xs)
    print("vs: ", vs)
    print("fn 2: ", fn2(xs))
    print("gradient of fn 2: ", grad2(xs))
    print("inv grad of grad of fn 2 should just be abs xs: ", inv_grad2(grad2(xs)))
    print("hess mult of vs: ", hess_mult2(xs, vs))
    print("inv hess mult of hess mult of vs should just be vs: ",
            hess_inv_mult2(xs, hess_mult2(xs, vs)))

if __name__ == "__main__":
    test_f1()
    test_f2()
