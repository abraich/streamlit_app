
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()



# install grf package if not installed
try:
    grf = rpackages.importr('grf')

grf = importr('grf')

codeR = """
n <- 2000
p <- 10
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- make.names(1:p)

W <- rbinom(n, 1, 0.4 + 0.2 * (X[, 1] > 0))
Y <- pmax(X[, 1], 0) * W + X[, 2] + pmin(X[, 3], 0) + rnorm(n)
cf <- causal_forest(X, Y, W)

"""

r = ro.r
