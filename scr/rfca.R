if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

if(!require(pcalg)){install.packages('pcalg')}
BiocManager::install("RBGL")
BiocManager::install("graph")

require(pcalg)

##################################################
## Example with hidden variables
## Zhang (2008), Fig. 6, p.1882
##################################################
## create the graph g
p <- 4
L <- 1 # '1' is latent
V <- c("Ghost", "Max","Urs","Anna","Eva")
edL <- setNames(vector("list", length=length(V)), V)
edL[[1]] <- list(edges=c(2,4),weights=c(1,1))
edL[[2]] <- list(edges=3,weights=c(1))
edL[[3]] <- list(edges=5,weights=c(1))
edL[[4]] <- list(edges=5,weights=c(1))
g <- new("graphNEL", nodes=V, edgeL=edL, edgemode="directed")
## compute the true covariance matrix of g
cov.mat <- trueCov(g)
## delete rows and columns belonging to latent variable L
true.cov <- cov.mat[-L,-L]

## transform covariance matrix into a correlation matrix
true.corr <- cov2cor(true.cov)
## The same, for the following three examples
indepTest <- gaussCItest
suffStat <- list(C = true.corr, n = 10^9)
## find PAG with FCI algorithm.
## As dependence "oracle", we use the true correlation matrix in
## gaussCItest() with a large "virtual sample size" and a large alpha:
normal.pag <- rfci(suffStat, indepTest, alpha = 0.9999, labels = V[-L],
                  verbose=TRUE)


#questions: how to extract adjacent matrix
# how to add input data


data("gmL")
suffStat <- list(C = cor(gmL$x), n = nrow(gmL$x))
fci.gmL <- rfci(suffStat, indepTest=gaussCItest, alpha = 0.9999, labels = c("2","3","4","5"))



########################## 
#Real Code
#Raquel AOki