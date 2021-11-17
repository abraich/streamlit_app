
class Scheme:
    """
    input:
        type: 'linear' or 'nonlinear'
        function: function to be used
    """
    def __init__(type,function):
        self.type = type
        self.function = function
    def __call__(self,x):
        return self.function(x)
    
        


def S(self, xbeta_p, tt_p, t):  # Agathe : pourquoi X apparait et pas Xbeta ?
        c_tt = self.coef_tt * tt_p  # coef_tt : coef de traitement

        if self.scheme == 'linear':
            return np.exp(-((self.lamb * t) ** self.alpha) * np.exp(xbeta_p + c_tt))
        else:
            sh_z = xbeta_p + np.cos(xbeta_p + c_tt) + \
                np.abs(xbeta_p - c_tt) + c_tt
            return np.exp(-((self.lamb * t) ** self.alpha) * np.exp(sh_z))