library(tidyverse)
library(ranger)
library(parallel)
library(pROC)

setwd('F:/DataScientist/revisions_data')
list.files()

db=read.csv('weatherAUS.csv', stringsAsFactors = T ) %>%
    mutate(Y=RainTomorrow)%>%
    select(-c(RainTomorrow))%>%
    filter(!is.na(Y)) %>% 
    filter(!is.na(Cloud9am)) %>% 
    filter(!is.na(Cloud3pm))

# ==============================================================================
# Equilibrage ?
# ==============================================================================

db %>% 
    count(Y) %>% 
    mutate(prop=n/sum(n))

# Assez déséquilibré. Il faudrait run smote. 


index=sample(1:nrow(db),nrow(db)*0.8)
train=db[index,]
test=db[-index,]

index_valid=sample(1:nrow(train),nrow(train)*0.2)
valid=train[index_valid,]
train=train[-index_valid,]

## Pour XGBoost
## XXtrain = model.matrix(Y~.,train)
## XXvalid = model.matrix(Y~.,valid)
## XXtest  = model.matrix(Y~.,test)


## =============================================================================
## Hyperparamétrage $ 1. ntrees 2. mtry 3. depth 4. sampling 5. split
## =============================================================================

hyper_grid = expand.grid(
    'trees'=c(500,1000),
    'max_depth'=c(1,3,5,8),
    'mtry'=c(1,5,10),
    'oob.err'=0
)
hyper_grid

p=1
for(p in 1:nrow(hyper_grid)){
        tmp=ranger(Y~., data=train,
              num.trees=hyper_grid[p,'trees'],
              num.threads = 0,
              mtry=hyper_grid[p,'mtry'],
              importance='impurity',
              oob.error = T,
              max.depth=hyper_grid[p,'max_depth'])
    
        hyper_grid$oob.err[p] = tmp$prediction.error  # Stockage direct, pas de `p=p+1`
}

library(ggplot2)
ggplot(hyper_grid %>% filter(trees==500))+
    aes(x=max_depth,y=oob.err)+
    geom_line()+
    facet_wrap(~mtry)+
    ggtitle('Hyper-paramétrage forêt: Erreur out-of-bag')+
    theme_bw()+
    ylab('')+
    xlab('Profondeur d\'arbre')



hyper_grid
hyper_grid[which.min(hyper_grid$oob.err),]

best_params <- hyper_grid[which.min(hyper_grid$oob.err), ]



final_model <- ranger(
    Y ~ ., 
    data = train,  
    num.trees = best_params$trees,
    mtry = best_params$mtry,
    importance = "impurity",
    probability=F,                           # Nécessaire pour tracer la ROC
    max.depth = best_params$max_depth
)

# ==============================================================================
# Prédictions 
# ==============================================================================
predictions <- predict(final_model, data = valid)
pred_class=predictions$predictions
# Si je veux ma matrice de confusion , il faut setup proba = F dans ranger()
conf=table(PRED=pred_class,OBS=valid$Y)

conf[2,2]

accuracy = sum(diag(conf))/sum(conf) ; accuracy
TP = conf[2,2] ; FN = conf[1,2] ; sensi = TP/(TP+FN) ; sensi
FP = conf[2,1]; prec= TP / (TP + FP); prec


# Compute Precision
Precision <- TP / (TP + FP)

# Si je veux ma roc, il faut setup proba = T dans ranger()
proba = predictions$predictions[,2]    # Je chope ma proba de prédire 1 
roc_obj=roc(valid$Y,proba)

ggplot(data.frame(fpr = roc_obj$specificities, tpr = roc_obj$sensitivities), aes(x = 1 - fpr, y = tpr)) +
    geom_line(color = "blue", size = 1) +
    geom_abline(linetype = "dashed", color = "red") +  # Diagonal reference line
    theme_minimal() +
    labs(title = paste("ROC Curve (AUC =", round(auc(roc_obj), 3), ")"),
         x = "False Positive Rate (1 - Specificity)",
         y = "True Positive Rate (Sensitivity)")


# If it's a classification problem:



