#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#
#Author: Raquel AOki
#January 2020 
#----------#----------#----------#----------#----------#----------#----------#
#----------#----------#----------#----------#----------#----------#----------#

#Plots 

rm(list=ls())

RUN_F1_SCORE = FALSE
RUN_CAUSAL_ROC = TRUE

if(RUN_F1_SCORE){
  #Comparing F1 score
  dt = read.table("C:\\Users\\raque\\Documents\\GitHub\\project_spring2019\\results\\experiments1.txt", header =T, sep = ';')
  old = c('adapter','lr','OneClassSVM','random','randomforest','upu')
  new = c('Adapter PU', 'Logistic Regression','One-class SVM','Random','Random Forest','Unbiased PU')
  dt$model_name = as.character(dt$model_name)
  for(i in 1:length(old)){
    dt$model_name[dt$model_name==old[i]]=new[i]
  }
  table(dt$model_name)
  
  #increase size, work on the colors and dots
  ggplot(data=dt, aes(x=f1, y=f1_, group=model_name, color=model_name)) + 
    geom_line() + geom_point()+
    scale_color_brewer(palette="Paired")+
    theme_minimal()+
    xlab('F1-score Testing set')+ylab('F1-score Full Set')+
    theme(legend.position="bottom")+
    labs(color = "Model")
  
  dt$ninnout = as.character(dt$nin)
  dt$ninnout[dt$nin=='[all]' & dt$nout=='[]']='Complete Data Only'
  dt$ninnout[dt$nin=='[FEMALE, MALE]' & dt$nout=='[]']='Gender Data Only'
  dt$ninnout[dt$nin=='[]' & dt$nout=='[all, FEMALE, MALE]']='Cancer Type Data Only'
  dt$ninnout[dt$nin=='[all, FEMALE, MALE]' & dt$nout=='[]']='Complete and Gender Data'
  dt$ninnout[dt$nin=='[all]' & dt$nout=='[FEMALE, MALE]']='Complete and Cancer Type Data'
  dt$ninnout[dt$nin=='[FEMALE, MALE]' & dt$nout=='[all]']='Gender and Cancer Type Data'
  dt$ninnout[dt$nin=='[]' & dt$nout=='[]']='Complete, Gender and Cancer Type Data'
  
  random = max(dt$f1[dt$model_name=='Random'])
  best = subset(dt,f1>random)
  best = best[order(best$f1, decreasing = TRUE),]
  
  require(xtable)
  #top25 = best[1:25,]
  #xtable(table(dt$ninnout,dt$data_name))
  #xtable(table(top25$ninnout,top25$model_name))
  
  top = rbind(subset(best,model_name=='Unbiased PU')[1:4,],
  subset(best,model_name=='Adapter PU')[1:3,],
  subset(best,model_name=='Logistic Regression')[1:3,])
  
  
  top0 = subset(top, select = c(ninnout,data_name,model_name))
  rownames(top0) = NULL
  xtable(top0)
}

if(RUN_CAUSAL_ROC){
  require(readxl)
  require(ggplot2)
  require(plotROC)
  require(gridExtra)
  #https://cran.r-project.org/web/packages/plotROC/vignettes/examples.html
  setwd("~/GitHub/project_spring2019/results")
  #create a list of the files from your target directory
  file_list <- list.files(path="~/GitHub/project_spring2019/results")
  
  flag = TRUE
  count = 0
  #had to specify columns to get rid of the total column
  for (i in 1:length(file_list)){
    file = file_list[i]
    if( strsplit(file,'_')[[1]][1] == 'roc'){
      if(flag){
        data = read.table(file, sep = ';', header = T)  
        ksize =  strsplit(file,'_')[[1]][2]
        if(ksize == 'mf10' || ksize == 'pca10'|| ksize == 'ac10'){
          data$k = 10
          data$method = gsub('10','',ksize)
          data$id = file
        }
        if(ksize == 'mf20' || ksize == 'pca20'|| ksize == 'ac20'){
          data$k = 20
          data$method = gsub('20','',ksize)
          data$id = file
        }
        if(ksize == 'mf40' || ksize == 'pca40'|| ksize == 'ac40'){
          data$k = 40
          data$method = gsub('40','',ksize)
          data$id = file
          
        }
        if(ksize == 'mf60' || ksize == 'pca60'|| ksize == 'ac60'){
          data$k = 60
          data$method = gsub('60','',ksize)
          data$id = file
          
        }
        if(ksize == 'bart'){
          data$pred = 1-data$pred
          data$k = 30 
          data$method = ksize
          data$id = file
          
        }
        flag = FALSE
        count = 1
      }else{
          data0 = read.table(file, sep = ';', header = T)  
          ksize =  strsplit(file,'_')[[1]][2]
          if(ksize == 'mf10' || ksize == 'pca10'|| ksize == 'ac10'){
            data0$k = 10
            data0$method = gsub('10','',ksize)
            data0$id = file
            
          }
          if(ksize == 'mf20' || ksize == 'pca20'|| ksize == 'ac20'){
            data0$k = 20
            data0$method = gsub('20','',ksize)
            data0$id = file
          }
          if(ksize == 'mf40' || ksize == 'pca40'|| ksize == 'ac40'){
            data0$k = 40
            data0$method = gsub('40','',ksize)
            data0$id = file
          }
          if(ksize == 'mf60' || ksize == 'pca60'|| ksize == 'ac60'){
            data0$k = 60
            data0$method = gsub('60','',ksize)
            data0$id = file
          }
          if(ksize == 'bart'){
            data0$pred = 1-data0$pred
            data0$k = 30 
            data0$method = ksize
            data0$id = file
          }
          data = rbind(data,data0)
          count = count + 1
      }

    }
  }
  require(wesanderson)
  data$k = as.factor(data$k)
  data0 = subset(data, method == 'bart' )
  data1 = subset(data, method ==  'ac' )
  data2 = subset(data, method ==  'mf' )
  data3 = subset(data, method == 'pca' )
  g0 <- ggplot(data0, aes(d = y01, m = pred, fill = id, color = k)) +  
    geom_roc(show.legend = FALSE,n.cuts = 0) + style_roc() + ggtitle('BART')+
    scale_color_brewer(palette = 'RdYlBu')
  
  
  g1 <- ggplot(data1, aes(d = y01, m = pred, fill = id, color = k)) + 
    geom_roc(show.legend = FALSE,n.cuts = 0) + style_roc()+ ggtitle('DA+Autoencoder')  +
    scale_color_brewer(palette = 'Oranges')
    #scale_color_manual(values=wes_palette(n=4, name="Zissou1"))
  
  g2 <- ggplot(data2, aes(d = y01, m = pred, fill = id, color = k)) + 
    geom_roc(show.legend = FALSE,n.cuts = 0) + style_roc()+ ggtitle('DA+Matrix Factorization')  +
    scale_color_brewer(palette = 'Oranges')
  
  
  g3 <- ggplot(data3, aes(d = y01, m = pred, fill = id, color = k)) + 
    geom_roc(show.legend = FALSE,n.cuts = 0) + style_roc()+ ggtitle('DA+PCA')  +
    scale_color_brewer(palette = 'Oranges')
  
  #+
  #  scale_color_manual(breaks = c("10", "20", "40",'60'),
  #                     values=c("#fcba03", "#e89510", "#e87510",'#e83f10'))

  
  
  grid.arrange(g0,g1,g2,g3, ncol=2)
}