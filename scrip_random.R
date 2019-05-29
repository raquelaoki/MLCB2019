rm(list=ls())
data = read.csv('C:\\Users\\raoki\\Documents\\GitHub\\project_spring2019\\Data\\seeds.csv', header=T)
head(data)

y = data[,1]
data = data[,-2]
data$X0 = as.factor(data$X0) 
fit = glm(X0~., data = data, family = 'binomial')
summary(fit)

y_pred = predict(fit,newdata = data[,-1], type = 'response' )
yp = y_pred
y_pred[y_pred>0.5] = 1
y_pred[y_pred<=0.5] = 0
table(y,y_pred)
table(y,y_pred)*100/length(y)



summary(fit$coef[-1])
summary(fit)
sd(fit$coef, na.rm =TRUE)