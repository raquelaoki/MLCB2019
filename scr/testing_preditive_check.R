mu = sample(1:100,100)
var = sample(1:10,100, replace = TRUE)
n = 5000
data = matrix(nrow=n,ncol=length(mu))
data = data.frame(data) 


for(i in 1:length(mu)){
  data[,i] = rnorm(n,mu[i],var[i])
}


model = prcomp(data,center = TRUE, scale. = TRUE)
latent = predict(model, data)[,1:20]
v = c()
v_r = c() #random 
for(i in 1:dim(data)[2]){
  xm = data[,i]
  split = sample(c(1:dim(data)[1]), size = dim(data)[1]*0.7,replace = F)
  
  xm_train = xm[split]
  xm_test = xm[-split]
  latent_train = latent[split,]
  latent_test = latent[-split,]
  
  latent_train_ = data_[split,]
  latent_test_ = data_[-split,]
  
  df = data.frame(xm_train, latent_train)
  df_ = data.frame(xm_train, latent_train_)
  
  model1 = lm(xm_train~., data = df)
  v[i] = sum(xm_test < predict(model1, data.frame(latent_test)))/length(xm_test)
  v_r[i] = sum(xm_test < mean(xm_train))/length(xm_test)
}


par(mfrow= c(1,2))
hist(v, col = 'lightgreen')
hist(v_r,col = 'lightgreen')


#t.test(v,mu=0.5)
#t.test(v_r,mu=0.5)

summary(v)
summary(v_r)
