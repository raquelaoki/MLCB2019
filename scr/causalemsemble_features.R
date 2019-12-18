#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#
#Author: Raquel AOki
#December 2019 
#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#

#Description: Given dataset, use BART and GFCI as causal methods 
rm(list=ls())


#----------#----------#----------#----------#----------#----------#----------#
#BART 
#----------#----------#----------#----------#----------#----------#----------#

options(java.parameters = "-Xmx5g")
library(bartMachine)
#https://cran.r-project.org/web/packages/bartMachine/vignettes/bartMachine.pdf
#recommended package BayesTrees is not functional anymore 

setwd("~/GitHub/project_spring2019")
data = read.table('data/tcga_train_gexpression_cgc_7k.txt', sep = ';', header = T)
extra = data[,c(1,2,3)]
data = data[,-c(1,2,3)]
y = as.factor(extra$y)

bart_machine = bartMachine(data, y, num_trees = 50, num_burn_in = 500, num_iterations_after_burn_in = 1400 )
summary(bart_machine)

#checking convergence 
plot_convergence_diagnostics(bart_machine)


obs = predict(bart_machine, data)
#making the interventional data, one for each gene 
dif = data.frame(gene = names(data),mean=c(rep(999, dim(data)[2])), sd=c(rep(999, dim(data)[2])))
for(v in 1:dim(data)[2]){
  data_v = data
  data_v[,v] = 0
  fit = predict(bart_machine, data_v)
  dif$mean[v] = mean(obs-fit)
  dif$sd[v] = sd(obs-fit)
}
write.table(dif,'results\\bart.txt', sep = ";", row.names = FALSE)

#----------#----------#----------#----------#----------#----------#----------#
#GFCI 
#----------#----------#----------#----------#----------#----------#----------#
rm(list = ls())

Sys.setenv(JAVA_HOME='C:/Program Files (x86)/Java/jre1.8.0_231/')
Sys.setenv(JAVA_HOME='C:/Program Files/Java/jre1.8.0_231/')
#install.packages("rJava")
#install.packages("stringr")
#install_github("bd2kccd/r-causal")

require(rJava)
require(stringr)
library(devtools)
library(rcausal)


########################## 
#Toy example
########################## 

#data("charity")   #Load the charity dataset

#tetradrunner.getAlgorithmDescription(algoId = 'fges')
#tetradrunner.getAlgorithmParameters(algoId = 'fges',scoreId = 'fisher-z')
#Compute FGES search
#tetradrunner <- tetradrunner(algoId = 'fges',df = charity,scoreId = 'fisher-z',
#                             dataType = 'continuous',alpha=0.1,faithfulnessAssumed=TRUE,maxDegree=-1,verbose=TRUE)

#tetradrunner$nodes #Show the result's nodes
#tetradrunner$edges #Show the result's edges

#Source
#https://bd2kccd.github.io/docs/r-causal/

########################## 
#Real Code
#Raquel AOki
########################## 

setwd("~/GitHub/project_spring2019")
data = read.table('data/tcga_train_gexpression_cgc_7k.txt', sep = ';', header = T)
extra = data[,c(1,2,3)]
data = data[,-c(1,2,3)]

t = c(100, 400, 500, 600, 700,800)
test = c()
for(t0 in 1:length(t)){
  bd = data[,1:t[t0]]
  #tetradrunner.getAlgorithmDescription(algoId = 'gfci ')
  #tetradrunner.getAlgorithmParameters(algoId = 'gfci',scoreId = 'fisher-z', testID = "correlation-t")
  #Compute FGES search
  tetradrunner <- tetradrunner(algoId = 'gfci',df = bd,scoreId = 'fisher-z',
                               testID = 'fisher-z',
                               dataType = 'continuous',alpha=0.001,
                               faithfulnessAssumed=TRUE,maxDegree=5,verbose=FALSE)
  
  #testID tetradrunner.listIndTests()
  #algoID = tetradrunner.listAlgorithms()
  #scoreId tetradrunner.listScores()
  
  
  #head(tetradrunner$nodes) #Show the result's nodes
  #head(tetradrunner$edges) #Show the result's edges
  test[t0] = length(tetradrunner$edges)
}

df = data.frame(tetradrunner$edges)

df[,1] = as.character(df[,1])
write.table(df, file = 'results/example_edges.txt', row.names = FALSE, sep = ';')

#plot
library(DOT)
graph_dot <- tetradrunner.tetradGraphToDot(tetradrunner$graph)
dot(graph_dot)
