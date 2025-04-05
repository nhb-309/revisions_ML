library(xgboost)
library(tidyverse)

path = 'F:/DataScientist/revisions_data/'; setwd(path)
db_raw=read.table('abalone.data',sep=',') 

colnames(db_raw) = c('sex','length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight','rings')
db=db_raw %>% 
    filter(sex!='I') %>% 
    mutate(sex=case_when(sex=='M'~1,
                         sex=='F'~0)) %>% 
    rename(Y=sex)

# Préparation des données XGBOOST : 
# prendre le jeu: 1) passage au format as.matrix() 2) retirer la target
# créer une variable qui stocke le vecteur : y = iris$Species
# créer le xgb.DMatrix(data = )

indY = which(names(db)=='Y');indY
X = as.matrix(db[,-indY])
Y = db[,indY]
xgb_data=xgb.DMatrix(data = X, label = Y)

iterations = 300

xgrid = expand.grid(
    max_depth = c(1,2,5,10),
    eta = c(0.15,0.05,0.01),
    subsample = c(0.4,0.6,0.8),
    lambda = c(0,3),
    alpha = c(0,3)
)

results = data.frame()

for(pp in 1:nrow(xgrid)){
    print(pp/nrow(xgrid))
    params = list(
        objective = "binary:logistic",
        eval_metric = "logloss",
        max_depth = xgrid$max_depth[pp],
        eta = xgrid$eta[pp],
        subsample = xgrid$subsample[pp],
        lambda = xgrid$lambda[pp],
        alpha = xgrid$alpha[pp]
    )
    set.seed(123)
    tmp = xgb.cv(
        params = params,
        nrounds = iterations,
        nfold = 5,
        verbose = F,
        early_stopping_rounds = 10,
        data=xgb_data
    )
    bestIter = tmp$best_iteration
    bestLogloss = tmp$evaluation_log$test_logloss_mean[bestIter]

    x = data.frame(cbind(xgrid[pp,],bestIter,bestLogloss))
    results[pp,1:7] = x
    
};results

resultsDef=results%>% 
    mutate(pb_grille=case_when(bestIter == iterations ~ 'GRILLE',
                               TRUE~'OK'))

resultsDef[which.max(resultsDef$bestLogloss),] 


