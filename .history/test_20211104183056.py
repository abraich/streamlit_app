
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects.packages as rpackages

# import grf package from R


from rpy2.robjects.packages import importr
grf = importr('grf')

