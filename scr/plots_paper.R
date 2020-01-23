#PLOTS 

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
