
class Scheme:
    def __init__(type,function):
        self.type = 'linear'
        self.function = function
    def __call__(self,x):
        return self.function(x)
    
        
    