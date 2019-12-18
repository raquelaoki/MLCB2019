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

setwd("~/GitHub/project_spring2019")
data = read.table('data/tcga_train_gexpression_cgc_7k.txt', sep = ';', header = T)
extra = data[,c(1,2,3)]
data = data[,-c(1,2,3)]
y = as.factor(extra$y)

bart_machine = bartMachine(data, y, num_trees = 50, num_burn_in = 500, num_iterations_after_burn_in = 1500 )
summary(bart_machine)


pred1 = predict(bart_machine, data[data[,1]<=mean(data[,1]),])
mndiffs1 = mean(as.numeric(as.character(bart_machine$y_hat_train[data[,1]<=mean(data[,1])]))-as.numeric(as.character(pred1)))
mndiffs1 #catt or satt

pred2 = predict(bart_machine, data[data[,1]>mean(data[,1]),])
mndiffs2 = mean(as.numeric(as.character(bart_machine$y_hat_train[data[,1]>mean(data[,1])]))-as.numeric(as.character(pred2)))
mndiffs2

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
