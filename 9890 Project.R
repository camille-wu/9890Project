#install.packages('psych')
#install.packages('ggplot2')
#install.packages(('gridExtra'))
library(glmnet)
library(randomForest)
library(psych)
library(ggplot2)
library(gridExtra)
set.seed(1)


df = read.csv('/Users/fivesheep/Desktop/3rd Semester/9890/Project/futures.csv')
sdev = apply(df[,-1],2,sd)
df[,-1] = t(t(df[,-1])/sdev) # Standardizing predictors, divided by standard deviation.

head(df)
dim(df) # 982 x 50

n =      dim(df)[1] # 982
p =      dim(df)[2]-1 # 49
X =      data.matrix(df[,-1])
y =      (df[,1])
X.orig = X
describe(y)
hist(y, main='Histogram of future intraday return 
     (every 15 seconds)')

n.train        =     floor(0.8*n)
n.test         =     n-n.train

M              =     100 # repeat 100 times
Rsq.train.las  =     rep(0,M)
Rsq.test.las   =     rep(0,M)
Rsq.train.rid  =     rep(0,M)
Rsq.test.rid   =     rep(0,M)
Rsq.test.rf    =     rep(0,M)  
Rsq.train.rf   =     rep(0,M)
Rsq.test.en    =     rep(0,M)  
Rsq.train.en   =     rep(0,M)
timing.rid     =     rep(0,M)
timing.las     =     rep(0,M)
timing.en      =     rep(0,M)
timing.rf      =     rep(0,M)


for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  # fit ridge and calculate and record the train and test R squares 
  a=0 # ridge
  tic = Sys.time()
  cv.fit1           =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit1              =     glmnet(X.train, y.train, alpha = a, lambda = cv.fit1$lambda.min)
  timing.rid[m]     =     as.double(Sys.time() - tic)  
  y.train.hat1      =     predict(fit1, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat1       =     predict(fit1, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rid[m]   =     1-mean((y.test - y.test.hat1)^2)/mean((y - mean(y))^2)
  Rsq.train.rid[m]  =     1-mean((y.train - y.train.hat1)^2)/mean((y - mean(y))^2) 
  
   # fit lasso and calculate and record the train and test R squares 
  a=1 # lasso
  tic = Sys.time()
  cv.fit2           =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit2              =     glmnet(X.train, y.train, alpha = a, lambda = cv.fit2$lambda.min)
  timing.las[m]     =     as.double(Sys.time()-tic)
  y.train.hat2      =     predict(fit2, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat2       =     predict(fit2, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.las[m]   =     1-mean((y.test - y.test.hat2)^2)/mean((y - mean(y))^2)
  Rsq.train.las[m]  =     1-mean((y.train - y.train.hat2)^2)/mean((y - mean(y))^2)  
  
  # fit elastic-net and calculate and record the train and test R squares 
  a=0.5 # elastic-net
  tic = Sys.time()
  cv.fit3          =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit3             =     glmnet(X.train, y.train, alpha = a, lambda = cv.fit3$lambda.min)
  timing.en[m]     =     as.double(Sys.time()-tic)
  y.train.hat3     =     predict(fit3, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat3      =     predict(fit3, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]   =     1-mean((y.test - y.test.hat3)^2)/mean((y - mean(y))^2)
  Rsq.train.en[m]  =     1-mean((y.train - y.train.hat3)^2)/mean((y - mean(y))^2)  
  
  # fit RF and calculate and record the train and test R squares 
  tic = Sys.time()
  rf               =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  timing.rf[m]     =     as.double(Sys.time()-tic)
  y.test.hat4      =     predict(rf, X.test)
  y.train.hat4     =     predict(rf, X.train)
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat4)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat4)^2)/mean((y - mean(y))^2)  
  
  cat(sprintf("m=%3.f| Rsq.test.rid=%.2f,Rsq.test.las=%.2f,Rsq.test.rf=%.2f,  
              Rsq.test.en=%.2f| Rsq.train.rid=%.2f,Rsq.train.las=%.2f, 
              Rsq.train.rf=%.2f,Rsq.train.en=%.2f| \n", m,  
              Rsq.test.rid[m], Rsq.test.las[m], Rsq.test.rf[m], Rsq.test.en[m],  
              Rsq.train.rid[m], Rsq.train.las[m], Rsq.train.rf[m], Rsq.train.en[m]))
}

time.rid = mean(timing.rid) # 0.2033
time.las = mean(timing.las) # 0.1636
time.en  = mean(timing.en)  # 0.1685
time.rf  = mean(timing.rf)  # 4.4190

num.features.rid = colSums(fit1$beta != 0)
num.features.las = colSums(fit2$beta != 0)
num.features.en  = colSums(fit3$beta != 0)

# CV curves
plot(cv.fit1, main='Ridge')
plot(cv.fit2, main='Lasso')
plot(cv.fit3, main='Elastic-net')

# Boxplots of R2
boxplot(Rsq.train.rid,Rsq.train.las,Rsq.train.en,Rsq.train.rf, xlab='Train R2', names=c('rid','las','en','rf'))
boxplot(Rsq.test.rid,Rsq.test.las,Rsq.test.en,Rsq.test.rf, xlab='Test R2', names=c('rid','las','en','rf'))
print(describe(Rsq.train.rid), digits=4)
print(describe(Rsq.train.las), digits=4)
print(describe(Rsq.train.en), digits=4)
print(describe(Rsq.train.rf), digits=4)
print(describe(Rsq.test.rid), digits=4)
print(describe(Rsq.test.las), digits=4)
print(describe(Rsq.test.en), digits=4)
print(describe(Rsq.test.rf), digits=4)

# Residuals
r.train.rid = (y.train - y.train.hat1)[,1]
r.test.rid  = (y.test  - y.test.hat1)[,1]

r.train.las = (y.train - y.train.hat2)[,1]
r.test.las  = (y.test  - y.test.hat2)[,1]

r.train.en  = (y.train - y.train.hat3)[,1]
r.test.en   = (y.test  - y.test.hat3)[,1]

r.test.rf   = (y.test  - y.test.hat4)
r.train.rf  = (y.train - y.train.hat4)

# Boxplots of residuals
boxplot(r.train.rid,r.train.las,r.train.en,r.train.rf, xlab='Train Residuals', names=c('rid','las','en','rf'))
boxplot(r.test.rid,r.test.las,r.test.en,r.test.rf, xlab='Test Residuals', names=c('rid','las','en','rf'))

# Bootstrap
bootstrapSamples =     100
beta.rid.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.las.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples)
beta.rf.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.en.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)
timing.rid2      =     rep(0,M)
timing.las2      =     rep(0,M)
timing.en2       =     rep(0,M)
timing.rf2       =     rep(0,M)
Rsq.bs.las       =     rep(0,M)
Rsq.bs.en        =     rep(0,M)
Rsq.bs.rid       =     rep(0,M)
Rsq.bs.rf        =     rep(0,M) 

for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # fit bs rf
  tic              =     Sys.time()
  rf.bs            =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  timing.rf2[m]    =     as.double(Sys.time()-tic)
  beta.rf.bs[,m]   =     as.vector(rf$importance[,1])
  # fit bs rid
  a                =     0 # ridge
  tic              =     Sys.time()
  cv.bs.fit1       =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  bs.fit1          =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.bs.fit1$lambda.min)  
  timing.rid2[m]   =     as.double(Sys.time()-tic)
  beta.rid.bs[,m]  =     as.vector(bs.fit1$beta)

  # fit bs las
  a                =     1 # lasso
  tic              =     Sys.time()
  cv.bs.fit2       =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  bs.fit2          =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.bs.fit2$lambda.min) 
  timing.las2[m]   =     as.double(Sys.time()-tic)
  beta.las.bs[,m]  =     as.vector(bs.fit2$beta)
  # fit bs en
  a                =     0.5 # elastic-net
  tic              =     Sys.time()
  cv.bs.fit3       =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  bs.fit3          =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.bs.fit3$lambda.min)  
  timing.en2[m]    =     as.double(Sys.time()-tic)
  beta.en.bs[,m]   =     as.vector(bs.fit3$beta)
  cat(sprintf("Bootstrap Sample %3.f \n", m))
}

time.rid2 = mean(timing.rid2) # 0.2033
time.las2 = mean(timing.las2) # 0.1636
time.en2  = mean(timing.en2)  # 0.1685
time.rf2  = mean(timing.rf2)  # 4.4190

# calculate bootstrapped standard errors / alternatively you could use qunatiles to find upper and lower bounds
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")
rid.bs.sd   = apply(beta.rid.bs, 1, "sd")
las.bs.sd   = apply(beta.las.bs, 1, "sd")
en.bs.sd    = apply(beta.en.bs, 1, "sd")

# fit rf to the whole data
rf2               =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)
y.rf.hat          =     predict(rf2, X)
Rsq.bs.rf         =     1-mean((y - y.rf.hat)^2)/mean((y - mean(y))^2)
# fit rid to the whole data
a=0 # ridge
cv.fit4           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit4              =     glmnet(X, y, alpha = a, lambda = cv.fit4$lambda.min)
y.rid.hat         =     predict(fit4, newx = X, type = "response")
Rsq.bs.rid        =     1-mean((y - y.rid.hat)^2)/mean((y - mean(y))^2)
# fit las to the whole data
a=1 # lasso
cv.fit5           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit5              =     glmnet(X, y, alpha = a, lambda = cv.fit5$lambda.min)
y.las.hat         =     predict(fit5, newx = X, type = "response")
Rsq.bs.las        =     1-mean((y - y.las.hat)^2)/mean((y - mean(y))^2)
# fit en to the whole data
a=0.5 # elastic-net
cv.fit6           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit6              =     glmnet(X, y, alpha = a, lambda = cv.fit6$lambda.min)
y.en.hat          =     predict(fit6, newx = X, type = "response")
Rsq.bs.en         =     1-mean((y - y.en.hat)^2)/mean((y - mean(y))^2)

print(c(Rsq.bs.rid, Rsq.bs.las, Rsq.bs.en, Rsq.bs.rf))

betaS.rf               =     data.frame(names(X[1,]), as.vector(rf2$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")

betaS.rid              =     data.frame(names(X[1,]), as.vector(fit4$beta), 2*rid.bs.sd)
colnames(betaS.rid)    =     c( "feature", "value", "err")

betaS.las              =     data.frame(names(X[1,]), as.vector(fit5$beta), 2*las.bs.sd)
colnames(betaS.las)    =     c( "feature", "value", "err")

betaS.en               =     data.frame(names(X[1,]), as.vector(fit6$beta), 2*en.bs.sd)
colnames(betaS.en)     =     c( "feature", "value", "err")

# we need to change the order of factor levels by specifying the order explicitly.
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.rid$feature    =  factor(betaS.rid$feature, levels = betaS.rid$feature[order(betaS.rid$value, decreasing = TRUE)])
betaS.las$feature    =  factor(betaS.las$feature, levels = betaS.las$feature[order(betaS.las$value, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])

rfPlot2 =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

ridPlot2 =  ggplot(betaS.rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

lasPlot2 =  ggplot(betaS.las, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

enPlot2 =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

grid.arrange(ridPlot2, lasPlot2, enPlot2, rfPlot2, nrow = 2)



