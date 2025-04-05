library(FactoMineR)
library(car)
library(tidyverse)
library(glmnet)
library(pROC)
library(xgboost)
library(glm)
library(parsnip)
library(yardstick)
library(themis)

# https://www.tidymodels.org/find/parsnip/


path = 'F:/DataScientist/revisions_data/'
setwd(path);list.files()

db = read.csv('SAheart.csv',header = T) %>% 
    mutate(chd = case_when(chd == "Si" ~ 1,
                           chd == "No" ~ 0)) %>%
    mutate(chd = as.factor(chd)) %>% 
    rename(Y=chd) %>% 
    na.omit()

db %>% head()

db %>% count(Y)

db_split = initial_split(db,prop = 0.6,strata=Y)
db_train = training(db_split)
db_test  = testing(db_split)

db_recipe = recipe(Y~., data=db_train) %>% 
    step_scale(all_numeric_predictors()) %>% 
    step_dummy(all_nominal_predictors(), -all_outcomes()) %>% 
    step_smote(Y,neighbors = 5, over_ratio = 0.7, skip=T)

prepped_recipe <- prep(db_recipe, training = db_train)
baked_data <- bake(prepped_recipe, new_data = db)
baked_data %>% count(Y)


logistic_model = logistic_reg() %>% 
    set_engine('glm') %>% 
    set_mode('classification')
logistic_ridge_model = logistic_reg(penalty = 1, mixture = 0) %>% 
    set_engine('glmnet') %>% 
    set_mode('classification')
logistic_lasso_model = logistic_reg(penalty = 1, mixture = 1) %>% 
    set_engine('glmnet') %>% 
    set_mode('classification')
logistic_elnet_model = logistic_reg(penalty = tune(), mixture = tune()) %>% 
    set_engine('glmnet') %>% 
    set_mode('classification')



lm_wf = workflow() %>% 
    add_model(logistic_model) %>% 
    add_recipe(db_recipe)
log_ridge_wf = workflow() %>% 
    add_model(logistic_ridge_model) %>% 
    add_recipe(db_recipe)
log_lasso_wf = workflow() %>% 
    add_model(logistic_lasso_model) %>% 
    add_recipe(db_recipe)
log_elnet_wf = workflow() %>% 
    add_model(logistic_elnet_model) %>% 
    add_recipe(db_recipe)


cv_fold = vfold_cv(v=5, data=db_train,strata=Y)

grid = expand.grid(penalty = seq(1,10,by=0.5),
                   mixture = seq(0,1,by=0.1))

results = tune_grid(log_elnet_wf,
                    grid = grid,
                    resamples = cv_fold)


results %>% collect_metrics() %>% 
    arrange(desc(mean)) 

best_elnet = results %>% collect_metrics () %>% 
    filter(.metric == 'roc_auc') %>% 
    arrange(desc(mean)) %>% 
    slice(1)
best_elnet

best_elnet_model = logistic_reg(penalty=best_elnet$penalty, mixture = best_elnet$mixture) %>% 
    set_mode('classification') %>% 
    set_engine('glmnet')
best_elnet_wf = workflow() %>% 
    add_model(best_elnet_model) %>% 
    add_recipe(db_recipe)
###

log_fit = lm_wf %>% 
    fit_resamples(resamples = cv_fold,
                  metrics = metric_set(roc_auc))
log_ridge_fit = log_ridge_wf %>% 
    fit_resamples(resamples = cv_fold,
                  metrics = metric_set(roc_auc))
log_lasso_fit = log_lasso_wf %>% 
    fit_resamples(resamples = cv_fold,
                  metrics = metric_set(roc_auc))
best_elnet_fit = best_elnet_wf %>% 
    fit_resamples(resamples = cv_fold,
                  metrics = metric_set(roc_auc))

bind_rows(
log_fit %>% collect_metrics() %>% mutate('algo'='logistic_regression'),
log_ridge_fit %>% collect_metrics() %>% mutate('algo'='log_ridge_regression'),
log_lasso_fit %>% collect_metrics() %>% mutate('algo'='log_lasso_regression'),
best_elnet_fit %>% collect_metrics() %>% mutate('algo'='log_elnet_regression_best')
) %>% 
    arrange(desc(mean))
