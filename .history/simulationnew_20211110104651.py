
class Scheme:
    """
    input:
        type_s: 'linear' or 'nonlinear'
        function: function to be used
    """
    def __init__(self, type_s, function):
        self.type = type_s
        self.function = function  # S(xbeta,tt,y)
    
    def get_type_scheme(self):
        return self.type
    def get_function(self):
        return self.function
    
# test scheme
scheme = Scheme('linear', lambda x,t,y: x*t*y)
print(scheme.get_type_scheme())
