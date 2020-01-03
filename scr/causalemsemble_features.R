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
library(ROCR)
#https://cran.r-project.org/web/packages/bartMachine/vignettes/bartMachine.pdf
#recommended package BayesTrees is not functional anymore 

setwd("~/GitHub/project_spring2019")
data = read.table('data/tcga_train_gexpression_cgc_7k.txt', sep = ';', header = T)
data <- data[sample(nrow(data),replace=FALSE),]
#testing set
extra_test = data[1:round(dim(data)*0.3)[1],c(1,2,3)]
data_test = data[1:round(dim(data)*0.3)[1],-c(1,2,3)]
y_test = as.factor(extra_test$y)
#training set 
extra =  data[-c(1:round(dim(data)*0.3)[1]),c(1,2,3)]
data = data[-c(1:round(dim(data)*0.3)[1]),-c(1,2,3)]
y = as.factor(extra$y)

#fitting the BART model 
bart_machine = bartMachine(data, y, num_trees = 50, num_burn_in = 500, num_iterations_after_burn_in = 1400 )
summary(bart_machine)

#checking BART convergence 
plot_convergence_diagnostics(bart_machine)

#making predictions
pred_p = predict(bart_machine, data, type='prob') #returns the prob of being on label 1


#ROC saving some points 
def_roc <- function(pred_p,obs,prob){
  tp = 1-pred_p
  tp[tp<=prob] = 0 
  tp[tp>prob] = 1
  tp = factor(tp, levels=c(0,1))
  obs = factor(obs,levels = c(0,1))
  #Option 1
  #TP: 0,0/(0,0+0,1)
  #FP: 1,0/(1,0+1,1)
  #Option 2
  #TP: 1,1/(1,1+1,0)
  #FP: 0,1/(0,1+0,0)
  tab = table(tp,obs)
  #Prob, opt1_tp,opt1_fp,opt2_tp,opt2_fp
  output = c(prob,tab[1,1]/(tab[1,1]+tab[1,2]),tab[2,1]/(tab[2,1]+tab[2,2]),
                  tab[2,2]/(tab[2,2]+tab[2,1]), tab[1,2]/(tab[1,2]+tab[1,1]))
  return(output)
}
roc_data = rbind(def_roc(pred_p,y,0.01),def_roc(pred_p,y,0.02))
values = seq(0.03,1,by=0.01)
for(i in 1:length(values)){
  roc_data = rbind(roc_data,def_roc(pred_p,y,values[i]))
}
roc_data = data.frame(roc_data)
names(roc_data) = c('prob','tp1','fp1','tp2','fp2')
write.table(roc_data,'results\\roc_bart_all.txt', row.names = FALSE,sep=';')

#ROC CURVE PLOT
pred <- prediction(1-pred_p,y)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
plot(roc.perf, col = 'red',main='ROC')
abline(a=0, b= 1)
#is 0.5 the best? 

#looking for a threshold 
def_threshold <- function(obs,pred,th){
  #Using 0 as "Positive" example
  tp = 1-pred
  tp[tp<=th] = 0 
  tp[tp>th] = 1
  #tp = factor(tp,levels = c('1','0'))
  tp = as.factor(tp)
  
  tab = table(tp,obs)
  cat('Results for :',th,'\nTrue Positive')
  cat(tab[1,1]/(tab[1,1]+tab[1,2]))
  cat('\nFalse Positive')
  cat(tab[2,1]/(tab[2,1]+tab[2,2]))
  cat('\n')
  cat(tab[1,1],tab[1,2],tab[2,1],tab[2,2] ) 
  cat('\n')
  
}


def_threshold(y,pred_p,0.4)
def_threshold(y,pred_p,0.5)
def_threshold(y,pred_p,0.6)

#making the interventional data, one for each gene 
fit_test =  predict(bart_machine, data_test, type='prob')
dif = data.frame(gene = names(data),mean=c(rep(999, dim(data)[2])), 
                 sd=c(rep(999, dim(data)[2])), se = c(rep(999, dim(data)[2])))
#dif = read.table('results\\bart.txt', sep = ';', header=T)
for(v in 1:dim(data_test)[2]){
  data_v = data_test
  data_v[,v] = 0
  fit = predict(bart_machine, data_v)
  dif$mean[v] = mean(fit_test-fit)
  dif$sd[v] = sd(fit_test-fit)
  dif$se[v] = mean((fit_test-fit)^2)
}
write.table(dif,'results\\feature_bart_all.txt', sep = ";", row.names = FALSE)

#----------#----------#----------#----------#----------#----------#----------#
#GFCI - works on my laptop only because the java dependences
#----------#----------#----------#----------#----------#----------#----------#
rm(list = ls())

Sys.setenv(JAVA_HOME='C:/Program Files (x86)/Java/jre1.8.0_231')
Sys.setenv(JAVA_HOME='C:/Program Files/Java/jre1.8.0_231')

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
