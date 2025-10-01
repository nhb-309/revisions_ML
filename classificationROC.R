
#> Chargement des librairies de Machine Learning
library(caret)

library(glmnet)
library(ranger)
library(gbm)
library(rpart)

library(parallel)
library(pROC)

# Import
path = 'C:/Users/L830195/Documents/R/MachineLearning/datasets/saheart/'
setwd(path);list.files()

name = 'saheart.txt'

db = read.table(name,sep=',') %>% 
  rename(Y = V10) %>%
  mutate(Y = case_when(Y == 'No'~0,
                       Y == 'Si'~1)) %>% 
  mutate(V1 = as.numeric(V1),
         V2 = as.numeric(V2),
         V3 = as.numeric(V3),
         V4 = as.numeric(V4),
         V5 = as.factor(V5),
         V6 = as.numeric(V6),
         V7 = as.numeric(V7),
         V8 = as.numeric(V8),
         V9 = as.numeric(V9),
         ) %>% 
  na.omit()

db %>% str()

split = 0.8
id = sample(nrow(db),0.9*nrow(db))
train = db[id,]
test = db[-id,]

#> Scale toutes les variables numeriques
trainX = train[,!names(train)=='Y']
df_train = sapply(trainX[,sapply(trainX, is.numeric)],scale) %>% 
  as.data.frame() %>% 
  mutate(Y = train$Y)

#> On réintègre tous les facteurs

df_train$famhist = trainX[,sapply(trainX, is.factor)]

#> One hot encoding pour les variables catégorielles
#> 

dummy = dummyVars('~.',data= df_train)
df_train = data.frame(predict(dummy, newdata=df_train))

#> Passage au format matriciel

datX = model.matrix(Y~., data = df_train)[,-1]
datY = df_train$Y

#> Initialisation de la validation croisée
nCores = parallel::detectCores()-1

nbloc = 10 

RES=data.frame('Y'=df_train$Y,
               'logistic'=NA,
               'arbre'=NA,
               'foret'=NA,
               'gbm' = NA,
               'ridge'=NA,
               'lasso'=NA,
               'elnet'=NA)

folds = sample(rep(1:nbloc,length = nrow(df_train)))

for(ii in 1:nbloc){
  cat("Fold: ",ii,'\n')
  
  donA = df_train[folds != ii,]
  donT = df_train[folds == ii,]
  
  donXA = model.matrix(Y~., data= donA)[,-1]
  donXT = model.matrix(Y~., data= donT)[,-1]
  donYA = as.matrix(donA$Y)
  
  # logistic
  tmp = glm(Y~., data = donA, family = 'binomial')
  RES[folds == ii, 'logistic'] = predict(tmp, donT, type = 'response')
  cat(' ** [logistic] \n')
  
  # arbre
  tmp = rpart(Y~., data = donA)
  RES[folds == ii, 'arbre'] = predict(tmp, donT)
  cat(' ** [arbre] \n')
  
  # foret
  tmp = ranger(factor(Y)~., data = donA, num.trees = 200, mtry = 2, num.threads = nCores, probability = T)
  RES[folds == ii, 'foret'] = predict(tmp, donT, type = 'response')$predictions[,2]
  cat(' ** [foret] \n')
  
  # gbm
  tmp = gbm(Y~., data = donA, distribution = 'bernoulli', cv.folds = 5, n.trees = 200, interaction.depth = 2, shrinkage = 0.3, n.cores = nCores)
  best.iter = gbm.perf(tmp, method = 'cv')
  RES[folds == ii, 'gbm'] = predict(tmp, donT, n.trees = best.iter)
  cat(' ** [gbm] \n')
  
  # ridge  
  tmp = cv.glmnet(donXA, donYA, alpha = 0, nfolds = 5, family='binomial',type.measure = 'auc')
  RES[folds == ii, 'ridge'] = predict(tmp, newx = donXT, type = 'response', s='lambda.min')
  cat(' ** [ridge] \n')
  
  # lasso 
  tmp = cv.glmnet(donXA, donYA, alpha = 1, nfolds = 5, family='binomial',type.measure = 'auc')
  RES[folds == ii, 'lasso'] = predict(tmp, newx = donXT, type = 'response', s='lambda.min')
  cat(' ** [lasso] \n')
  
  # elnet
  tmp = cv.glmnet(donXA, donYA, alpha = 0.5, nfolds = 5, family='binomial',type.measure = 'auc')
  RES[folds == ii, 'elnet'] = predict(tmp, newx = donXT, type = 'response', s='lambda.min')
  cat(' ** [elnet] \n')  

}

RES[,sapply(RES, is.character)] = sapply(RES[,sapply(RES, is.character)],as.numeric) %>% as.data.frame()
RES %>% str()

# Receiver Operator Curves

roc_log   = roc(Y~logistic,data=RES)
roc_arb   = roc(Y~arbre, data=RES)
roc_foret = roc(Y~foret,data=RES)
roc_gbm   = roc(Y~gbm,data=RES)
roc_ridge = roc(Y~ridge,data=RES)
roc_lasso = roc(Y~lasso,data=RES)
roc_elnet = roc(Y~elnet,data=RES)

plotRocParams = data.frame('lwd' = 1.5,
                           'lty' = 'solid',
                           'legend.lwd' = 2.5,
                           'legend.cex' = 1)

par(bg = '#FFFDF0')
plot(roc_foret, col = '#009678',lty = plotRocParams$lty, main = 'Courbes ROC',lwd=plotRocParams$lwd)
lines(roc_log, col= '#756000', lty = plotRocParams$lty, lwd = plotRocParams$lwd)
lines(roc_gbm, col= '#CC5D47', lty = plotRocParams$lty,lwd = plotRocParams$lwd)
lines(roc_elnet,col = '#780096',lty =plotRocParams$lty,lwd = plotRocParams$lwd)
legend('bottomright',
       legend = 
         c(paste0('Forêt / AUC ', round(roc_foret$auc,2)),
           paste0('Elnet / AUC ',round(roc_ridge$auc,2)),
           paste0('GBM / AUC ', round(roc_gbm$auc,2)),
           paste0('Log / AUC ', round(roc_log$auc,2))),
       col = c('#009678','#780096','#CC5D47','#756000'),
       lty = c(plotRocParams$lty),
       lwd = plotRocParams$legend.lwd,
       cex = plotRocParams$legend.cex)

