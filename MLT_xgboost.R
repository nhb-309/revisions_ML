library(tidyverse)
library(xgboost)
library(MatrixModels)
setwd('F:/DataScientist/revisions_data/')

db = read.csv('SAheart.csv', stringsAsFactors = TRUE) %>%
    mutate(Y = case_when(chd == 'Si' ~ 1,    # "Si" becomes 1
                         chd == 'No' ~ 0)   # "No" becomes 0
    ) %>%
    mutate(Y=as.integer(Y)) %>% 
    select(-chd)

## =============================================================================
## Training Testing and Validation sets
## =============================================================================

## 1. Train / Test
index= sample(1:nrow(db),nrow(db)*0.8)
db_train=db[index,]
db_test=db[-index,]

## 2. Validation
# Further split db_train into train (80%) and validation (20%)
index_valid = sample(1:nrow(db_train), 0.2 * nrow(db_train))
db_valid = db_train[index_valid, ]
db_train = db_train[-index_valid, ]

## 3. Forme matrice --> retirer l'intercept
XX_train = model.matrix(Y ~ ., data = db_train)[, -1]
XX_valid = model.matrix(Y ~ ., data = db_valid)[, -1]
XX_test  = model.matrix(Y ~ ., data = db_test)[, -1]

## 4. Passage en DMatrix --> pour xgboost
dtrain = xgb.DMatrix(data = XX_train, label = db_train$Y)
dvalid = xgb.DMatrix(data = XX_valid, label = db_valid$Y)
dtest  = xgb.DMatrix(data = XX_test)


## =============================================================================
## Hyperparameters tuning 
## =============================================================================
eta_values = c(0.1, 0.01, 0.05, 0.2, 0.3)

# Initialize an empty data frame to store the results
results = data.frame(eta = numeric(), train_auc = numeric(), valid_auc = numeric())

# Loop over the values of eta

for (eta_val in eta_values) {
    # Define parameters with the current value of eta
    params = list(
        objective = "binary:logistic",
        eta = eta_val,
        max_depth = 1,  # You can also tune this hyperparameter
        eval_metric = "auc",
        lambda = 0    # Default value, you can change if needed
    )
    
    # Train the model with xgb.train (no cross-validation here)
    watchlist = list(train = dtrain, valid = dvalid)
    model = xgb.train(
        params = params,
        data = dtrain,
        nrounds = 6000,
        watchlist = watchlist,
        early_stopping_rounds = 20,
        verbose = 0
    )
     
    # Extract the AUC scores for training and validation
    model$evaluation_log
    train_auc = model$evaluation_log[which.max(model$evaluation_log$train_auc), "train_auc"]
    valid_auc = model$evaluation_log[which.max(model$evaluation_log$valid_auc), "valid_auc"]
    
    # Store the results in the results data frame
    results = rbind(results, data.frame(eta = eta_val, train_auc = train_auc, valid_auc = valid_auc))
    
    # Print the results of the current iteration
    print(paste("Completed eta =", eta_val, "| Train AUC =", round(train_auc, 4), "| Validation AUC =", round(valid_auc, 4)))
}

print(results)

# Find the best eta (highest validation AUC)
best_eta = results[which.max(results$valid_auc), ]
print(paste("Best eta:", best_eta$eta, "with Validation AUC:", round(best_eta$valid_auc, 4)))


XX=model.matrix(Y~., data=db_train)[,-1]
dbtrainx=xgb.DMatrix(XX, label=XX$Y)

tg=list(
    lrate=c(0.1,0.05,0.01,0.005),
    estopping=c(20,30,50,100),
    tdepth=c(1,3),
    lambda=c(0,1,2)
)
tg
tuning=expand.grid(
    'lrate'=NA,
    'estopping'=NA,
    'tdepth'=NA
)
tuning
p=1

curves=list()

for(t in 1:length(tg$lambda)){
    for(m in 1:length(tg$tdepth)){
        for(j in 1:length(tg$lrate)){
            for(k in 1:length(tg$estopping)){
                
                params=list(
                    objective = "binary:logistic",
                    max_depth=tg$tdepth[m],
                    eta=tg$lrate[j],
                    eval_metric='auc',
                    lambda=tg$lambda[t]
                )
                
                cv_results=xgb.cv(
                    params=params,
                    data=dbtrainx,
                    nfold=10,
                    nrounds=2000,
                    early_stopping_rounds = tg$estopping[k],
                    verbose=F
                )
                
                log=cv_results$evaluation_log
                curves[[p]]=log[,c('train_auc_mean','test_auc_mean')]
                
                tuning[p,'lrate']=tg$lrate[k]
                tuning[p,'estopping']=tg$estopping[j]
                tuning[p,'tdepth']=tg$tdepth[m]
                tuning[p,'lambda']=tg$lambda[t]
                
                tuning[p,'auc_test']=log[which.max(cv_results$evaluation_log$test_auc_mean),'test_auc_mean']
                tuning[p,'auc_train']=log[which.max(cv_results$evaluation_log$test_auc_mean),'train_auc_mean']
                tuning[p,'iter']=log[which.max(cv_results$evaluation_log$test_auc_mean),'iter']
                
                print(paste('Iteration = ',p,sep=''))
                
                p=p+1
                
            }
        }
    }
}

best=tuning[which.max(tuning$auc_test),] %>% rownames() %>% as.numeric()
tuning[best,]
tuning[which(tuning$tdepth==3 & tuning$lambda==1 & tuning$lrate==0.01),]
curves[[best]]



##> ============================================================================
##> Optimal xgboost model
##> ============================================================================
best_params=tuning[best,]
best_params
params=list(
    objective = "binary:logistic",
    max_depth=best_params$tdepth,
    eta=best_params$lrate,
    eval_metric='auc',
    lambda=best_params$lambda
)

cv_results=xgb.train(
    params=params,
    data=dbtrainx,
    nrounds=200,
    early_stopping_rounds = 20,
    verbose=T
)
cv_results


predict(cv_results,)
