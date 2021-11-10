import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
packageNames = ('grf', 'ggplot2')
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

packnames_to_install = [x for x in packageNames if not rpackages.isinstalled(x)]

# Running R in Python example installing packages:
if len(packnames_to_install) > 0:
    utils.install_packages(StrVector(packnames_to_install))
    
    
data = robjects.r('read.table(file = "http://personality-project.org/r/datasets/R.appendix3.data", header = T)')

# print data head
print(data.head())

Rcode = ''' n <- 2000
p <- 10
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- make.names(1:p)

W <- rbinom(n, 1, 0.4 + 0.2 * (X[, 1] > 0))
Y <- pmax(X[, 1], 0) * W + X[, 2] + pmin(X[, 3], 0) + rnorm(n)  '''


# define a class to use grf package to plot survival curves


