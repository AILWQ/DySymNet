"""Functions for use with symbolic regression.

These functions encapsulate multiple implementations (sympy, Tensorflow, numpy) of a particular function so that the
functions can be used in multiple contexts."""

import torch
# import tensorflow as tf
import numpy as np
import sympy as sp


class BaseFunction:
    """Abstract class for primitive functions"""

    def __init__(self, norm=1):
        self.norm = norm

    def sp(self, x):
        """Sympy implementation"""
        return None

    def torch(self, x):
        """No need for base function"""
        return None

    def tf(self, x):
        """Automatically convert sympy to TensorFlow"""
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'tensorflow')(x)

    def np(self, x):
        """Automatically convert sympy to numpy"""
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'numpy')(x)


class Constant(BaseFunction):
    def torch(self, x):
        return torch.ones_like(x)

    def sp(self, x):
        return 1

    def np(self, x):
        return np.ones_like


class Identity(BaseFunction):
    def __init__(self):
        super(Identity, self).__init__()
        self.name = 'id'

    def torch(self, x):
        return x / self.norm  # ??

    def sp(self, x):
        return x / self.norm

    def np(self, x):
        return np.array(x) / self.norm


class Square(BaseFunction):
    def __init__(self):
        super(Square, self).__init__()
        self.name = 'pow2'

    def torch(self, x):
        return torch.square(x) / self.norm

    def sp(self, x):
        return x ** 2 / self.norm

    def np(self, x):
        return np.square(x) / self.norm


class Pow(BaseFunction):
    def __init__(self, power, norm=1):
        BaseFunction.__init__(self, norm=norm)
        self.power = power
        self.name = 'pow{}'.format(int(power))

    def torch(self, x):
        return torch.pow(x, self.power) / self.norm

    def sp(self, x):
        return x ** self.power / self.norm


# class Sin(BaseFunction):
#     def torch(self, x):
#         return torch.sin(x * 2 * 2 * np.pi) / self.norm
#
#     def sp(self, x):
#         return sp.sin(x * 2 * 2 * np.pi) / self.norm

class Sin(BaseFunction):
    def __init__(self):
        super().__init__()
        self.name = 'sin'

    def torch(self, x):
        return torch.sin(x) / self.norm

    def sp(self, x):
        return sp.sin(x) / self.norm


class Cos(BaseFunction):
    def __init__(self):
        super(Cos, self).__init__()
        self.name = 'cos'

    def torch(self, x):
        return torch.cos(x) / self.norm

    def sp(self, x):
        return sp.cos(x) / self.norm


class Tan(BaseFunction):
    def __init__(self):
        super(Tan, self).__init__()
        self.name = 'tan'

    def torch(self, x):
        return torch.tan(x) / self.norm

    def sp(self, x):
        return sp.tan(x) / self.norm


class Sigmoid(BaseFunction):
    def torch(self, x):
        return torch.sigmoid(x) / self.norm

    # def tf(self, x):
    #     return tf.sigmoid(x) / self.norm

    def sp(self, x):
        return 1 / (1 + sp.exp(-20 * x)) / self.norm

    def np(self, x):
        return 1 / (1 + np.exp(-20 * x)) / self.norm

    def name(self, x):
        return "sigmoid(x)"


# class Exp(BaseFunction):
#     def __init__(self, norm=np.e):
#         super().__init__(norm)
#
#     # ?? why the minus 1
#     def torch(self, x):
#         return (torch.exp(x) - 1) / self.norm
#
#     def sp(self, x):
#         return (sp.exp(x) - 1) / self.norm

class Exp(BaseFunction):
    def __init__(self):
        super().__init__()
        self.name = 'exp'

    # ?? why the minus 1
    def torch(self, x):
        return torch.exp(x)

    def sp(self, x):
        return sp.exp(x)


class Log(BaseFunction):
    def __init__(self):
        super(Log, self).__init__()
        self.name = 'log'

    def torch(self, x):
        return torch.log(torch.abs(x) + 1e-6) / self.norm

    def sp(self, x):
        return sp.log(sp.Abs(x) + 1e-6) / self.norm


class Sqrt(BaseFunction):
    def __init__(self):
        super(Sqrt, self).__init__()
        self.name = 'sqrt'

    def torch(self, x):
        return torch.sqrt(torch.abs(x)) / self.norm

    def sp(self, x):
        return sp.sqrt(sp.Abs(x)) / self.norm


class BaseFunction2:
    """Abstract class for primitive functions with 2 inputs"""

    def __init__(self, norm=1.):
        self.norm = norm

    def sp(self, x, y):
        """Sympy implementation"""
        return None

    def torch(self, x, y):
        return None

    def tf(self, x, y):
        """Automatically convert sympy to TensorFlow"""
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'tensorflow')(x, y)

    def np(self, x, y):
        """Automatically convert sympy to numpy"""
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'numpy')(x, y)

    # def name(self, x, y):
    #     return str(self.sp)


class Product(BaseFunction2):
    def __init__(self, norm=0.1):
        super().__init__(norm=norm)
        self.name = '*'

    def torch(self, x, y):
        return x * y / self.norm

    def sp(self, x, y):
        return x * y / self.norm


class Plus(BaseFunction2):
    def __init__(self, norm=1.0):
        super().__init__(norm=norm)
        self.name = '+'

    def torch(self, x, y):
        return (x + y) / self.norm

    def sp(self, x, y):
        return (x + y) / self.norm


class Sub(BaseFunction2):
    def __init__(self, norm=1.0):
        super().__init__(norm=norm)
        self.name = '-'

    def torch(self, x, y):
        return (x - y) / self.norm

    def sp(self, x, y):
        return (x - y) / self.norm


class Div(BaseFunction2):
    def __init__(self):
        super(Div, self).__init__()
        self.name = '/'

    def torch(self, x, y):
        return x / (y + 1e-6)

    def sp(self, x, y):
        return x / (y + 1e-6)


def count_inputs(funcs):
    i = 0
    for func in funcs:
        if isinstance(func, BaseFunction):
            i += 1
        elif isinstance(func, BaseFunction2):
            i += 2
    return i


def count_double(funcs):
    i = 0
    for func in funcs:
        if isinstance(func, BaseFunction2):
            i += 1
    return i


default_func = [
    Product(),
    Plus(),
    Sin(),
]
