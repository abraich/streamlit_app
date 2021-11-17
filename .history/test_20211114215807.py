import sympy

# define a class to calculate integral of a function

class Integral:
    def __init__(self, f, a, b):
        self.f = f
        self.a = a
        self.b = b
        self.n = 100
        self.x = sympy.Symbol('x')
        self.h = (self.b - self.a) / self.n
        self.I = 0
        self.I_list = []
        