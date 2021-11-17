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
    
    def calculate(self):
        for i in range(self.n):
            self.I += self.f.subs(self.x, self.a + i * self.h) * self.h
        return self.I

# test the class

f = sympy.sin(sympy.pi * x)

a = 0
b = 1

I = Integral(f, a, b)
result = I.calculate()
print(result)
        