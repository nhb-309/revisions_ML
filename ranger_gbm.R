library(xgboost)
library(bestglm)
library(gbm)
library(parallel)
library(ranger)
cores=detectCores()

db=read.csv('weatherAUS.csv',stringsAsFactors = T) %>% 
    na.omit() %>% 
    mutate(Y=as.factor(RainTomorrow)) %>% 
    select(MinTemp, Sunshine,Temp9am,
           WindSpeed3pm,Humidity9am,
           Pressure9am,Cloud9am,Y) %>% 
    mutate(Y=case_when(Y=='No'~1,
                       Y=='Yes'~0)) 

index=sample(seq_len(nrow(db)),floor(0.8*nrow(db)))
train=db[index,]
test=db[-index,]
index_valid=sample(seq_len(nrow(train)),floor(0.8*nrow(train)))
train=train[index_valid,]
valid=train[-index_valid,]

## =============================================================================
## Gradient Boosting
## =============================================================================

hgrid=expand.grid(
    'nt'=c(2000),
    'lr'=c(0.2,0.1,0.05),
    'depth'=c(1,2,3)
)
hgrid

p=1
for(p in 1:nrow(hgrid)){
    print(paste('### Iteration - ',p,'###'))
    m1=gbm(Y~.,
           data=train,
           distribution='bernoulli',
           n.trees=hgrid$nt[p],
           shrinkage=hgrid$lr[p],
           interaction.depth=hgrid$depth[p],
           cv.folds=4,
           n.cores=cores-1,
           verbose = F)
    best_iter = gbm.perf(m1)
    loglik_loss=m1$cv.error[best_iter]

    hgrid[p,'loglik_loss']=loglik_loss
    hgrid[p,'bestiter']=gbm.perf(m1)
    pred=predict(m1,newdata = valid ,type='response')
    hgrid[p,'auc']=roc(valid$Y,pred)$auc %>% round(3)
}

hgrid[which.max(hgrid$auc),]

final_gbm=gbm(Y~.,
              data=train,
              distribution='bernoulli',
              n.trees=2000,
              shrinkage=0.05,
              interaction.depth=3,
              cv.folds=4,
              n.cores=cores,
              verbose = F)


## =============================================================================
## Random Forest avec ranger
## ============================================================================= 

rf_hgrid=expand.grid(
    'trees'=c(500),
    'mtry'=c(3,7),
    'depth'=c(1,3,5),
    'alpha'=c(0,0.5,1),
    'regul_factor'=c(seq(0.00001,1,0.25))
)

for(p in 1:nrow(rf_hgrid)){
    print(paste('### Iteration - ',p,'###'))
    t=system.time({
        m3=ranger(Y~., data=train,
                  probability=TRUE,
                  # HP Forêt
                  num.trees = rf_hgrid$trees[p],
                  mtry=rf_hgrid$mtry[p],
                  max.depth=rf_hgrid$depth[p],
                  min.node.size = 10,
                  # Pénalisation
                  alpha = rf_hgrid$alpha[p],
                  regularization.factor = rf_hgrid$regul_factor[p]
                  # Parallélisation
                  #num.threads = cores --> pas de parallelisation en régul
        )
        
        pred=predict(m3,data=valid,type='response')$predictions[,2]
        rf_hgrid[p,'auc']=roc(valid$Y,pred)$auc %>% round(5)    
        
    }
    )
    rf_hgrid[p,'runtime']=t['elapsed']
}
 

rf_hgrid
rf_hgrid[which.max(rf_hgrid$auc),]

final_rf=ranger(Y~.,
                data=train,
                probability=TRUE,
                num.trees=500,
                mtry=3,
                max.depth=5,
                alpha=1,
                regularization.factor = 0.25)









