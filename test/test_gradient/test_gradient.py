import numpy as np
import theano
import theano.tensor as T


def test_gradient():
    a = theano.shared(np.arange(6).reshape(-1, 2))
    c = T.repeat(a, [2, 4, 5], axis=0)
    T.grad(c[0, 0], a)


def test_gradient_1d():
    a = theano.shared(np.arange(6).reshape(-1, 2))
    x = T.scalar()
    c = T.repeat(a**x, 5, axis=0)
    g = T.grad(c[5, 1], x)
    f = g.eval({x:5})
    print(f)
