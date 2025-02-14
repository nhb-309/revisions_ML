
library(tidyverse)
library(glmnet)
library(pROC)
setwd('F:/DataScientist/revisions_data/')
db=read.csv('weatherAUS.csv',stringsAsFactors = T) %>% 
    na.omit() %>% 
    mutate(Y=as.factor(RainTomorrow)) %>% 
    select(MinTemp, Sunshine,Temp9am,
           WindSpeed3pm,Humidity9am,
           Pressure9am,Cloud9am,Y) %>% 
    mutate(Y=case_when(Y=='No'~1,
                       Y=='Yes'~0))

index=sample(seq_len(nrow(db)), size= floor(0.8*nrow(db)))

train = db[index, ]
test  = db[-index,]

train_X= model.matrix(Y~., data=train)[,-1]
test_X = model.matrix(Y~., data=test)[,-1]

train_Y=train$Y
test_Y =test$Y

ridge=cv.glmnet(train_X,train_Y,alpha = 0 ,  family='binomial', nfolds=10, type.measure='class')
lasso=cv.glmnet(train_X,train_Y,alpha = 1,   family='binomial', nfolds=10, type.measure='class')
elast=cv.glmnet(train_X,train_Y,alpha = 0.5, family='binomial', nfolds=10, type.measure='class')

pridge = predict(ridge, newx=test_X, type='response', s='lambda.min')
plasso = predict(lasso, newx=test_X, type='response', s='lambda.min')
pelast = predict(elast, newx=test_X, type='response', s='lambda.min')

roc.ridge=roc(test_Y,as.vector(pridge))
roc.lasso=roc(test_Y,as.vector(plasso))
roc.elast=roc(test_Y,as.vector(pelast))

par(mfrow=c(1,3))
plot.roc(roc.ridge)
plot.roc(roc.lasso,col='red')
plot.roc(roc.elast, col='blue')

##
## Matrice de confusion
##

opt_thresh_ridge = coords(roc.ridge, "best", best.method='youden')$threshold
opt_thresh_lasso = coords(roc.lasso, "best", best.method='youden')$threshold
opt_thresh_elast = coords(roc.elast, "best", best.method='youden')$threshold

opt_thresh_ridge
opt_thresh_lasso
opt_thresh_elast

pred_ridge = ifelse(pridge > opt_thresh_ridge,1, 0)
pred_lasso = ifelse(plasso > opt_thresh_lasso,1, 0)
pred_elast = ifelse(pelast > opt_thresh_elast,1, 0)

confridge=table(Predict= pred_ridge, Actual = test_Y)
conflasso=table(Predict= pred_lasso, Actual = test_Y)
confelast=table(Predict= pred_elast, Actual = test_Y)

# True Positive Rate 
confridge
TP=confridge[2,2]
FN=confridge[1,2]
TP/(TP+FN)

# True Negative Rate
TN=confridge[1,1]
FP=confridge[2,1]
TN/(TN+FP)

