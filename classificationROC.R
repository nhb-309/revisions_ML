path = 'C:/Users/L830195/Documents/R/MachineLearning/datasets/abalone/'
setwd(path)
list.files()

name = 'abalone.data'

db = read.table(name,sep=',') %>% 
  mutate(V1 = case_when(V1 == 'M'~0,
                        V1 == 'F'~1)) %>% 
  rename(Y = V1) %>% 
  na.omit()

split = 0.95
id = sample(nrow(db),0.9*nrow(db))
train = db[id,]
test = db[-id,]

#> Scale toutes les variables numeriques
trainX = train[,!names(train)=='Y']
df_train = sapply(trainX[,sapply(trainX, is.numeric)],scale) %>% 
  as.data.frame() %>% 
  mutate(Y = train$Y)

#> One hot encoding pour les variables catégorielles
#> ici: pas de variable catégorielle

#> Passage au format matriciel

datX = model.matrix(Y~., data = df_train)[,-1]
datY = df_train$Y

#> Chargement des librairies de Machine Learning
library(glmnet)
library(ranger)
library(parallel)
library(pROC)

#> Initialisation de la validation croisée
nCores = parallel::detectCores()-1

nbloc = 6

RES=data.frame('Y'=df_train$Y,
               'foret'=NA,
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
  

  
  # foret
  tmp = ranger(factor(Y)~., data = donA, num.trees = 500, mtry = 4, num.threads = nCores, probability = T)
  RES[folds == ii, 'foret'] = predict(tmp, donT, type = 'response')$predictions[,2]
  cat(' ** [foret] \n')
  
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

roc_foret = roc(Y~foret,data=RES)
roc_ridge = roc(Y~ridge,data=RES)
roc_lasso = roc(Y~lasso,data=RES)
roc_elnet = roc(Y~elnet,data=RES)

rocCV = data.frame(
  "alg" = c('foret','ridge','lasso','elnet'),
  "auc" = c(roc_foret$auc,roc_ridge$auc, roc_lasso$auc, roc_elnet$auc))
rocCV
rocCV[which.max(rocCV$auc),]

plotRocParams = data.frame('lwd' = 1,
                           'lty' = 'dotted',
                           'legend.lwd' = 3.5,
                           'legend.cex' = 1)

par(bg = '#FAFAEB')
plot(roc_foret, col = '#009678',lty = 'dotted', main = 'Courbes ROC',lwd=plotRocParams$lwd)
lines(roc_elnet,col = '#780096',lty = 'dotted',lwd = plotRocParams$lwd)
legend('bottomright',
       legend = 
         c(paste0('Forêt / AUC ', round(roc_foret$auc,2)),
           paste0('Elnet / AUC ',round(roc_ridge$auc,2))),
       col = c('#009678','#780096'),
       lty = c(plotRocParams$lty),
       lwd = plotRocParams$legend.lwd,
       cex = plotRocParams$legend.cex)

