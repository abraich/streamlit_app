import sympy

# define a class to calculate integral of a function

class Integral:
    def __init__(self, f):
        self.f = f
        self.x = sympy.Symbol('x')
        self.integral = sympy.integrate(self.f, self.x)
       
    
    def get_integral(self):
        return self.integral
    # get latex representation of the integral
    def get_latex(self):
        return sympy.latex(self.integral)
    

# test the class
if __name__ == '__main__'
    f = sympy.sin(x)
    integral = Integral(f)
    print(integral.get_integral())
    print(integral.get_latex())
    
    