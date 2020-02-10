#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#
#BART
#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#

#Description: Given dataset, use BART and GFCI as causal methods
rm(list=ls())

RUN_BART = TRUE
RUN_CATE = FALSE
RUN_ALL = FALSE
RUN_ = TRUE

#ROC saving some points
def_roc <- function(pred_p,obs,prob){
  tp = 1-pred_p
  tp[tp<=prob] = 0
  tp[tp>prob] = 1
  tp = factor(tp, levels=c(0,1))
  obs = factor(obs,levels = c(0,1))

  tab = table(tp,obs)
  #Prob, opt1_tp,opt1_fp,opt2_tp,opt2_fp
  output = c(prob,tab[1,1]/(tab[1,1]+tab[1,2]),tab[2,1]/(tab[2,1]+tab[2,2]),
             tab[2,2]/(tab[2,2]+tab[2,1]), tab[1,2]/(tab[1,2]+tab[1,1]))
  return(output)
}

#----------#----------#----------#----------#----------#----------#----------#
#BART
#----------#----------#----------#----------#----------#----------#----------#
if(RUN_BART&RUN_ALL){
  options(java.parameters = "-Xmx5g")
  library(bartMachine)
  library(ROCR)
  #Reference
  #https://cran.r-project.org/web/packages/bartMachine/vignettes/bartMachine.pdf

  setwd("~/GitHub/project")
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

  roc_data = data.frame(pred_p, y)
  names(roc_data) = c('pred','y01')

  write.table(roc_data,'results\\roc_bart_all.txt', row.names = FALSE,sep=';')

  if(RUN_CATE){
  #making the interventional data, one for each gene
  fit_test =  predict(bart_machine, data_test, type='prob')
  dif = data.frame(gene = names(data),mean=c(rep(999, dim(data)[2])),
                   sd=c(rep(999, dim(data)[2])), se = c(rep(999, dim(data)[2])))
  for(v in 1:dim(data_test)[2]){
    data_v = data_test
    data_v[,v] = 0
    fit = predict(bart_machine, data_v)
    dif$mean[v] = mean(fit_test-fit)
    dif$sd[v] = sd(fit_test-fit)
    dif$se[v] = mean((fit_test-fit)^2)
  }
  write.table(dif,'results\\feature_bart_all.txt', sep = ";", row.names = FALSE)
  }
}


if(RUN_BART&RUN_){
  options(java.parameters = "-Xmx5g")
  library(bartMachine)
  library(ROCR)
  set_bart_machine_num_cores(4) #new
  #Reference
  #https://cran.r-project.org/web/packages/bartMachine/vignettes/bartMachine.pdf

  setwd("~/GitHub/project_spring2019")
  files = read.table('data/files_names.txt', sep = ';', header = T)
  files$files = paste('data/',files$files, sep='')
  files$ci = as.character(files$ci)
  files$class = as.character(files$class)

  for(f in 1:dim(files)[1]){
    data = read.table(files$files[f], sep = ';', header = T)
    if(dim(data)[1]>=100){
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
    #summary(bart_machine)

    #checking BART convergence
    #plot_convergence_diagnostics(bart_machine)

    #making predictions
    pred_p = predict(bart_machine, data, type='prob') #returns the prob of being on label 1


    roc_data = data.frame(pred_p, y)
    names(roc_data) = c('pred','y01')
    write.table(roc_data,paste('results\\roc_bart_',files$ci[f],'_',files$class[f],'.txt',sep=''), row.names = FALSE,sep=';')

    #making the interventional data, one for each gene
    if(RUN_CATE){

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
    write.table(dif,paste('results\\feature_bart_',files$ci[f],'_',files$class[f],'.txt',sep=''), sep = ";", row.names = FALSE)
    }

  }
}
}
