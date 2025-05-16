install.packages('caret') # necessaire pour la validation croisee stratifiee et le cv
install.packages('tidyverse')
install.packages('pROC')
install.packages('gbm')
install.packages('ranger')
install.packages('glmnet')
install.packages('xgboost')
install.packages('tidymodels')
install.packages('recipes')
install.packages('bestglm')
install.packages('kernlab')

library(bestglm)
library(caret) # necessaire pour la validation croisee stratifiee et le cv
library(tidyverse)
library(pROC)
library(gbm)
library(ranger)
library(glmnet)
library(xgboost)
library(tidymodels)
library(recipes)
library(bestglm)
library(kernlab)

ncores = parallel::detectCores()

# Répertoire de travail
setwd("F:/DataScientist/revisions_ML/")

db = SAheart %>% 
    rename(Y=chd) %>% 
    mutate(Y=as.factor(Y))

# lecture
db <- read.csv('bank.csv',sep=';')  %>% 
    rename(Y=y) %>%
    mutate(Y = case_when(Y=='no'~0,
                         Y=='yes'~1)) %>% 
    mutate(Y=as.factor(Y))

# visualisation rapide
db %>% 
  summary()

db %>% str()

# on n'a pas de valeurs manquantes

# Premières transformations ----

# la variable cible est "y" qui est indiquée comme étant "yes" ou "no"
# on recode toutes les variables caractères en facteurs
# on centre-réduit les prédicteurs numériques

#db <- db %>% 
#  mutate(y=ifelse(y=="no",0,1))

# ICI, ATTENTION SI UN FACTEUR EST CODE EN NUMERIQUE ORDONNE
my_recipe = recipe(Y ~ ., data = db) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_predictors())

don <- my_recipe %>% 
  prep() %>% 
  bake(new_data = db)

#don <- db %>% 
#  rename(Y=y) %>% 
#    mutate(Y=as.factor(Y))

# Comparaison d'algo ----
# Division en Apprentissage et Test
split=0.9

trainSize=round(0.9*nrow(don))

A = sample(nrow(don), trainSize)

donApp  = don[A,]

donTest = don[-A,]

# Division en Apprentissage et Validation

nbloc=10

folds = createFolds(donApp$Y,
                    k=nbloc,
                    list=F) # a voir si createFolds garde la proportion automatiquement

# Pre-processing
donAppX = model.matrix(Y~., data=donApp)
donY = donApp$Y

# ==============================================================================
# Grilles
# ==============================================================================

# Foret
gr.foret=expand.grid(num.trees=c(100),
                     max.depth=c(1,5))

gr.foret.params=data.frame(gr.foret,'auc'=NA)

# SVM
gr.poly=expand.grid(C=c(0.1,10,100),
                    degree=c(1,2,3),
                    scale=1)
gr.radial=expand.grid(C=c(0.1,1,10),
                      sigma = c(0.0001,0.001,0.01,0.1,1))

ctrl = trainControl(method='cv', number=5)


# Validation croisee
SCORE=data.frame('Y'=donApp$Y,'logistic'=NA,'aic'=NA,'bic'=NA,'ridge'=NA,'lasso'=NA,'elnet'=NA,'foret'=NA,'rad_svm'=NA,'pol_svm'=NA,'gbm'=NA,'xgb'=NA)
SCORE %>% head()
SCORE %>% dim()

jj=1

for(jj in 1:nbloc){
  
  cat('Fold: ', jj, '\n')
  
  donA=donApp[folds!=jj,]
  donV=donApp[folds==jj,]
  
  donXA=donAppX[folds!=jj,]
  donXV=donAppX[folds==jj,]  
  donYA=donY[folds!=jj]
  
  
  # Logistique =================================================================
  logistic=glm(Y~., data=donA, family='binomial')
  SCORE[folds==jj,'logistic'] = predict(logistic,newdata=donV,type='response')
  
  # AIC ========================================================================
  aic=stats::step(logistic,trace=0)
  SCORE[folds==jj,'aic']=predict(aic,newdata=donV,type='response')
 
  ## BIC ========================================================================
  bic=stats::step(logistic,trace=0,k=log(nrow(donA)))
  SCORE[folds==jj,'bic']=predict(bic,newdata=donV,type='response')
  
  # Penalisation =================================================================
  
  ridge=cv.glmnet(donXA,donYA,alpha=0  ,family='binomial',nfolds=5,type.measure='auc')
  lasso=cv.glmnet(donXA,donYA,alpha=1  ,family='binomial',nfolds=5,type.measure='auc')
  elnet=cv.glmnet(donXA,donYA,alpha=0.5,family='binomial',nfolds=5,type.measure='auc')
  SCORE[folds==jj,'ridge']=predict(ridge,newx=donXV,type='response',s='lambda.min')
  SCORE[folds==jj,'lasso']=predict(lasso,newx=donXV,type='response',s='lambda.min')
  SCORE[folds==jj,'elnet']=predict(elnet,newx=donXV,type='response',s='lambda.min')
  
  # Foret ======================================================================
  
  ### Hyper-parametrage
  
  control <- trainControl(method="cv",number=5)
  
  best_params <- data.frame()
  
  for (j in 1:nrow(gr.foret)) {
      cat('Foret - ', j, '\n')
      tuned_model <- train(Y~.,data=donA,
                           method="ranger",
                           classification = T,
                           metric = 'Accuracy',
                           trControl=control,
                           tuneGrid=expand.grid(
                               mtry=c(1,3),
                               splitrule='gini',
                               min.node.size=c(1,3)),
                           num.trees=gr.foret$num.trees[j],
                           max.depth=gr.foret$max.depth[j])
      
      results = tuned_model$results
      
      params = results[which.max(results$Accuracy),] %>% 
          mutate(num.trees=gr.foret$num.trees[j],
                 max.depth=gr.foret$max.depth[j])
      
      best_params = rbind(best_params, params)
  }
  
  param_optimaux = best_params[which.max(best_params$Accuracy),]
  
  foret.finale=ranger(factor(Y)~., data=donA,
                      classification=T, probability = T,
                      num.trees=param_optimaux$num.trees,
                      max.depth = param_optimaux$max.depth,
                      min.bucket = param_optimaux$min.buckets,
                      mtry = param_optimaux$mtry)
  
  SCORE[folds==jj,'foret'] = predict(foret.finale,data=donV,type='response')$predictions[,'1']
  
  
  
  # SVM ========================================================================
  #svm.poly = train(Y~., data=donA, method = 'svmPoly', trControl = ctrl, tuneGrid = gr.poly)
  #bestPoly=svm.poly$results[which.max(svm.poly$results$Accuracy),]
  #tmpPol=ksvm(Y~., data=donA, kernel = 'polydot', kpar=list(degree=bestPoly$degree,scale=1,offset=1),C=bestPoly$C,prob.model=T)
  #
  #svm.radial = train(Y~., data=donA, method = 'svmRadial', trControl = ctrl, tuneGrid = gr.radial)
  #bestRadial=svm.radial$results[which.max(svm.radial$results$Accuracy),]
  #tmpRad=ksvm(Y~., data=donA, kernel =  'rbfdot', kpar=list(sigma=bestRadial$sigma),C=bestRadial$C,prob.model=T)
  #
  #SCORE[folds==jj,'pol_svm'] = predict(tmpPol, newdata=donV, type = 'prob')[,'1'] # rajouter l'index de colonne : il ne faut en prendre qu'une
  #SCORE[folds==jj,'rad_svm'] = predict(tmpRad, newdata=donV, type= 'prob')[,'1']  
  
 
  ## =============================================================================
  ## Gradient Boosting
  ## =============================================================================
  # 1. Prepare data: ensure binary response is 0/1
  donA.gbm = donA %>% mutate(Y = as.numeric(Y) - 1)
  
  tmp <- gbm(Y~.,data=donA.gbm,distribution = "bernoulli",cv.folds=5,n.trees=15,
             interaction.depth = 3,shrinkage = 0.1)
  best.iter <- gbm.perf(tmp, method = "cv")
  SCORE[folds==jj,"gbm"] = predict.gbm(tmp,donV,type='response',n.trees = best.iter)
  
  # XGBOOST =====================================================================
  
  indY = which(names(donA)=='Y')
  
  indY
  
  X_train = as.matrix(donA[,-indY])
  
  Y_train = as.numeric(donA[[indY]])-1
  
  X_test = as.matrix(donV[,-indY])
  
  Y_test = as.numeric(donV[[indY]])-1
  
  xgb_data_train=xgb.DMatrix(data = X_train, label = Y_train)
  
  xgb_data_test=xgb.DMatrix(data = X_test, label = Y_test)
  
  xgrid = expand.grid(
    max_depth = c(1,2,5),
    eta = c(0.1,0.05)
  )
  
  results = data.frame()
  
  
  # AJOUTER LE EARLY STOPPING ROUND ET UN BORD DE GRILLE:
  #> ok pour le early stopping round
  #> 
  for(pp in 1:nrow(xgrid)){
    
    cat('\n',100*pp/nrow(xgrid),'% \n')
    
    params = list(
      objective = "binary:logistic",
      eval_metric = "logloss",
      max_depth = xgrid$max_depth[pp],
      eta = xgrid$eta[pp]
    )
    
    nIter = 600
    
    tmp = xgb.cv(
      params = params,
      nrounds = nIter,
      nfold = 5,
      verbose = F,
      early_stopping_rounds = 10,
      data=xgb_data_train
    )
      
    bestIter = tmp$best_iteration
    
    #if(bestIter==nIter){
    #  cat('\n',"ATTENTION: BORD DE GRILLE: ", bestIter, '\n')
    #} else{
    #    cat('\n' ,'Meilleure itération XGBoost : ', bestIter, '\n' )
    #}
    
    bestLogloss = tmp$evaluation_log$test_logloss_mean[bestIter]
    
    x = data.frame(cbind(xgrid[pp,],bestIter,bestLogloss))
    
    results[pp,1:nrow(xgrid)] = x
  }
  
  best_params_index <- results[which.max(results$bestLogloss),] 
  
  checkIter = best_params_index$bestIter
  
  if(checkIter == nIter){
      cat('\n', "ATTENTION: BORD DE GRILLE --> ", checkIter)
  }else{
      cat('\n','Meilleure itération XGBoost: ', checkIter)
  }
  
  best_params = list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = best_params_index$max_depth,
    eta = best_params_index$eta
  )
  
  xgb = xgb.train(params = best_params,data = xgb_data_train, nrounds = best_params_index$bestIter)
  
  SCORE[folds==jj, 'xgb']=  predict(xgb,newdata = xgb_data_test,type='response')
  
}

# ==============================================================================
# > Comparaison des modèles
# ==============================================================================


SCORE

rocCV = roc(factor(Y)~., data=SCORE %>% select(-c(rad_svm,pol_svm)), quiet=T)

aucmodele = sort(round(unlist(lapply(rocCV,auc)),5),dec=TRUE)

tmp = lapply(rocCV, FUN = function(r) {
    co = coords(r, x = "best",
                ret = c('threshold','tp','fp','fn','tn','sensitivity','specificity','accuracy'),
                transpose = TRUE)
    
    # Compute F1 score
    tp = co["tp"]
    fp = co["fp"]
    fn = co["fn"]
    f1 = if ((2 * tp + fp + fn) == 0) NA else 2 * tp / (2 * tp + fp + fn)
    
    # Add F1 to the coords result
    co["f1"] <- f1
    return(co)
})
mat=do.call(rbind,tmp)

aucmodele
mat

# Modèle le plus performant : 
# Selon quel critère : 


# REESTIMER LE MODELE SUR l'ENSEMBLE DES DONNES ET PREDIRE LE TEST
# faire le feature engineering

# Meilleur modèle => xgboost donc on le récupère:


# AJOUTER LE EARLY STOPPING ROUND ET UN BORD DE GRILLE:
#> ok pour le early stopping round

# ==============================================================================
# > Entraînement du meilleur modèle. 
# ==============================================================================

### 0. Sélection de tout le jeu d'apprentissage

indY = which(names(donApp)=='Y')
X_train_final = as.matrix(donApp[,-indY])
Y_train_final = as.numeric(donApp[[indY]])-1

xgb_data_final_train = xgb.DMatrix(data=X_train_final, label = Y_train_final)
getinfo(xgb_data_final_train,'label') # pour vérifier la composition du label (en 0/1)

### 1. Grille d'hyperparamètres

metric = 'logloss'  # ou 'logloss' --> penser à modifier dans la boucle d'HP

xgrid = expand.grid(
    max_depth = c(1,2,5),
    subsample = c(0.3,0.5,0.9),
    eta = c(0.1,0.05,0.01)
)


### 2. Boucle hyper-paramètres
pp=1
for(pp in 1:nrow(xgrid)){
    
    cat('\n',100*pp/nrow(xgrid),'% \n')
    
    params = list(
        objective = "binary:logistic",
        eval_metric = metric,
        max_depth = xgrid$max_depth[pp],
        eta = xgrid$eta[pp]
    )
    
    nIter = 900
    
    tmp = xgb.cv(
        params = params,
        nrounds = nIter,
        nfold = 5,
        verbose = F,
        early_stopping_rounds = 10,
        data=xgb_data_train
    )
    
    bestIter = tmp$best_iteration
    bestLogloss = tmp$evaluation_log$test_logloss_mean[bestIter]

    if(bestIter==nIter){
      cat('\n',"ATTENTION: BORD DE GRILLE: ", bestIter, '\n')
    } else{
        cat('\n' ,'Meilleure itération XGBoost : ', bestIter, '\n' )
    }
    
    x = data.frame(cbind(xgrid[pp,],bestIter,bestLogloss))
    
    xgrid[pp,'bestIter'] = bestIter  
    xgrid[pp,'bestLogloss'] = bestLogloss

}

xgrid[which.min(xgrid$bestLogloss),!names(xgrid)%in% c('bestLogloss')]
final.params = xgrid[which.min(xgrid$bestLogloss),!names(xgrid)%in% c('bestLogloss')]

nrounds = final.params$bestIter

xgb_final = xgb.train(objective = 'binary:logistic',
                      data = xgb_data_final_train,
                      eta = final.params$eta,
                      subsample = final.params$subsample,
                      nrounds = nrounds)



# Test et performances sur le dataset de test final

X_test_final = as.matrix(donTest[,-indY])
Y_test_final = as.numeric(donTest[[indY]])-1
xgb_data_final_test = xgb.DMatrix(data=X_test_final, label = Y_test_final)

SCORE_final=data.frame('Y'=donTest$Y,'xgb'=NA)

SCORE_final[,'xgb']=predict(xgb_final, newdata=xgb_data_final_test, type='response')
SCORE_final

rocCV = roc(factor(Y)~., data=SCORE_final, quiet=T)

auc(rocCV)


#===============================================================================
# > Feature engineering
#=============================================================================== 

### 1. Polynomes

carre = function(x){
    return(x^2)
}
cube = function(x){
    return(x^3)
}

index.num.app = sapply(donApp, is.numeric)
index.num.app

donApp.squared = sapply(donApp[,index.num.app],carre)
colnames(donApp.squared) = paste0('sqd_',colnames(donApp.squared))
donApp.cubed   = sapply(donApp[,index.num.app],cube)
colnames(donApp.cubed) = paste0('cub_',colnames(donApp.cubed))

donTest.squared = sapply(donTest[,index.num.app], carre)
colnames(donTest.squared) = paste0('sqd_',colnames(donTest.squared))
donTest.cubed = sapply(donTest[,index.num.app],cube)
colnames(donTest.cubed) = paste0('cub_',colnames(donTest.cubed))


donApp.poly = data.frame(cbind(donApp,donApp.squared, donApp.cubed))
donTest.poly = data.frame(cbind(donTest,donTest.squared, donTest.cubed))

mat.donApp.poly = model.matrix(Y~., data= donApp.poly) 
donApp.Y = as.numeric(donApp$Y) -1

mat.donTest.poly = model.matrix(Y~., data=donTest.poly)

xgb.data.poly = xgb.DMatrix(data=mat.donApp.poly, label = donApp.Y)

xgb.test.poly = xgb.DMatrix(data=mat.donTest.poly)
donTest.Y = as.numeric(donTest$Y)-1

# 
SCORE.poly = data.frame('Y' = donTest.Y)
#

nIter = 10

params = list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 3,
    eta = 0.1
)

model.poly <- xgb.train(
    params = params,
    data = xgb.data.poly,
    nrounds = nIter,
    verbose = 0
)

xgb_final = xgb.train(objective = 'binary:logistic',
                      eval_metric = 'logloss',
                      data = xgb.data.poly,
                      eta = final.params$eta,
                      subsample = final.params$subsample,
                      nrounds = nrounds)





SCORE.poly[,'predict_poly']=predict(model.poly, newdata = xgb.test.poly, type='response')

rocCV = roc(factor(Y)~., data=SCORE.poly, quiet=T)

auc(rocCV)







