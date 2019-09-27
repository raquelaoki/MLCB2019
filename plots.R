require(ggplot2)
require(gridExtra)

#Reading files
b_ac = read.table("baselines_acc.txt")
b_f1 = read.table("baselines_f1.txt")
p_ac = rbind(read.table("pgm_id12_acc.txt"),read.table("pgm_id12_acc_mylaptop.txt"))
p_f1 = rbind(read.table("pgm_id12_f1.txt"),read.table("pgm_id12_f1_mylaptop.txt"))


stats <- function(metric, name){
  cat('\n',name,'\n')
  metric$V1 = as.numeric(as.character(metric$V1))
  hist(metric$V1,main=name, col='lightgreen')
  metric = metric$V1[metric$V1>summary(metric$V1)[2]]
  #metric = metric$V1
  cat('\n summary: Min/1/median/mean/3/max\n',summary(metric))  
  error <- qt(0.975,df=length(metric)-1)*sd(metric)/sqrt(length(metric))
  cat('\n CI 95% - ', name, ' ',mean(metric)-error,mean(metric)+error)
}

stats(b_ac, 'baseline - ac')
stats(p_ac, 'PGM - ac')
stats(b_f1, 'baseline - f1')
stats(p_f1, 'PGM - f1')


data = data.frame(name = c(rep('Accuracy',dim(b_ac)[1]),rep('Accuracy',dim(p_ac)[1]),
                           rep('F1 Score',dim(b_f1)[1]),rep('F1 Score',dim(p_f1)[1])),
                  values = c(as.numeric(b_ac$V1),as.numeric(p_ac$V1),as.numeric(b_f1$V1),as.numeric(p_f1$V1)),
                  Model = c(rep('Baseline',dim(b_ac)[1]),rep('Model Combined',dim(p_ac)[1]),
                           rep('Baseline',dim(b_f1)[1]),rep('Model Combined',dim(p_f1)[1])))

data1 = subset(data,name=='Accuracy')
data2 = subset(data,name!='Accuracy')


g1<- ggplot(data1, aes(x=Model, y=values)) + geom_boxplot(fill='#3cb800')+ 
  xlab('')+ylab('Accuracy')
g2<- ggplot(data2, aes(x=Model, y=values)) + geom_boxplot(fill='#3cb800')+ 
  xlab('')+ylab('F1 Score')

grid.arrange(g1, g2, nrow = 1)

