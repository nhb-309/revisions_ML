# Load required libs - when you come to run the models you might get prompts to install the model packages if they aren't already
library(tidyverse)  # contains dplyr, ggplot and our data set
library(tidymodels) # the main tidymodels packages all in one place
library(embed)      # use the create embeddings for categorical features
library(stacks)     # used for stacking models 
library(rules)      # contains the cubist modelling algorithm
library(vip)        # variable importance plots
library(finetune)   # package for more advanced hyperparameter tuning  
library(doParallel) # parallelisation package     
library(brulee)     # torch model
library(tictoc)     # measure how long processes take
library(skimr)
set.seed(12345)

## Lien tutorial

https://www.stepbystepdatascience.com/ml-with-tidymodels

#########################################################


## =====================================================
## Pre-processing & data cleaning
## =====================================================

head(diamonds)
summary(diamonds)

diamonds_copy=
    diamonds %>% 
    mutate(carat=ifelse(carat>3, NA, carat),
           x=ifelse(x==0 | x>10, NA, x),
           y=ifelse(y==0 | y>10, NA, y),
           z=ifelse(z==0 | z>10, NA, z))

summary(diamonds_copy)

for (i in 1:4){
    column <-tibble(sample(1:100, nrow(diamonds_copy), replace=T))
    names(column) <- paste0("random", i)
    diamonds_copy <- cbind(diamonds_copy, column)
}

# Simulate some low variance features to test removing redundant features
diamonds_copy <- 
    diamonds_copy %>% 
    mutate(low_var = ifelse(sample(1:nrow(diamonds_copy), replace=F) <= nrow(diamonds_copy)/100, 1, 0)) 



## =====================================================
## Train / Test split
## =====================================================

train_test_split = initial_split(diamonds_copy,
                                 prop=0.1,
                                 strata=price)

train=training(train_test_split)
test= testing(train_test_split)

skim(train)

## =====================================================
## Processing & plots 
## =====================================================

categorical_vars = train %>% 
    select(!where(is.numeric),low_var) %>% 
    names()
continuous_vars = train %>% 
    select(where(is.numeric),-price) %>% 
    names()

plot_categorical <- function(i){
    ggplot(train %>% 
               mutate(across(low_var, ~as.factor(.x))), 
           aes_string(x=i, y= "price", color=i)) + # aes_string as the column name is being passed by a function
        geom_boxplot() + 
        ggtitle(paste0("Box plots of ", i)) +
        theme_light()
}
map(categorical_vars, plot_categorical)

## =====================================================
## Correlations within the data
## =====================================================


cor(train %>% 
        select(continuous_vars) %>% 
        filter(if_all(everything(),~!is.na(.x))))



## =====================================================
## Recettes
## =====================================================

base_rec=recipe(formula=price~.,data=train) 
base_rec

role_example=train %>%
    mutate(ID=row_number()) # copie train avec une col ID

role_example_rec=recipe(formula=price~., data=role_example)
role_example_rec

role_example_rec %>% 
    update_role(ID,new_role='ID')


recipe(formula=price~., data=train) %>% 
    step_dummy(all_nominal_predictors()) %>%     # dummy vars for factors
    step_normalize(all_numeric_predictors()) %>% # normalize numeric inputs 
    step_log(price)                              # log transformation on target


recipe(formula=price ~ . , data=train) %>% 
    step_impute_bag(x,y,z,carat) %>% 
    prep(train) %>% 
    bake(train) %>% 
    complete.cases() %>% 
    sum()/nrow(train)


recipe(formula=price~., data=train) %>% 
    step_corr(all_numeric_predictors()) %>% # plans to remove highly corr num predictors
    prep(train) %>%  # calcul des correlations entre prédicteurs numériques
    bake(train)      # suppression des variables trop corrélées

recipe(formula=price~., data=train) %>% 
    step_nzv(all_predictors()) %>% 
    prep(train) %>% 
    bake(train)

## 
## resampling
## 

recipe(formula = price~., data=train) %>% 
    step_impute_bag(carat,x,y,z) %>% 
    step_normalize(carat,x,y,z) %>% 
    step_pca(carat,x,y,z,threshold=0.9) %>% 
    prep(train) %>% 
    bake(train)

train_bootstraps = 
    bootstraps(train,times=25,strata=price)
train_bootstraps

train_cv=
    vfold_cv(train, v=10, repeats=5, strata=price)
train_cv

train_loo_cv=loo_cv(train)

diamond_groups =
    rbind(train %>% mutate(ID=row_number(), year=1),
          train %>% mutate(ID=row_number(), year=2),
          train %>% mutate(ID=row_number(), year=3),
          train %>% mutate(ID=row_number(), year=4),
          train %>% mutate(ID=row_number(), year=5)) %>% 
    mutate(price= price*(1.05^year)) %>% 
    arrange(ID,year)

head(diamond_groups,5)


train_group_vfold_cv =
    group_vfold_cv(diamond_groups,group=ID,v=5,repeats=3)


show_engines('linear_reg')


glmnet_model = 
    linear_reg(mode='regression',
               engine='glmnet',
               penalty=0.0001,
               mixture=1)
glmnet_model

### 

recipe <-
    recipe(formula = price ~ . , data=train) %>% 
    step_impute_bag(carat, x,y,z) %>%      # impute missing values
    step_unorder(color, cut, clarity) %>%  # remove the ordering from the factors
    step_dummy(color, cut, clarity) %>%    # create dummy variables
    step_normalize(all_numeric_predictors()) %>% # normalise all the numeric predictors
    step_nzv(all_predictors()) %>%   # remove any near zero variance features
    step_log(price) # log the target
recipe

lm_model=linear_reg(mode='regression') %>% 
    set_engine('lm')

lm_workflow <-
    workflow() %>% 
    add_recipe(recipe) %>% # add in our newly created recipe
    add_model(lm_model) # add the model spec we made earlier
lm_workflow

fit_lm=
    fit_resamples(lm_workflow,
                  resamples = train_cv)

collect_metrics(fit_lm)
show_best(fit_lm)
collect_notes(fit_lm)
collect_predictions(fit_lm)

fit_lm_mae=
    fit_resamples(lm_workflow,
                  resamples=train_cv,
                  control=control_resamples(
                      save_pred=T,
                      save_workflow=T,
                      verbose=T
                  ),
                  metrics = metric_set(mae,rsq))

show_best(fit_lm_mae,metric='mae')
collect_predictions(fit_lm_mae)

collect_predictions(fit_lm_mae) %>% 
    filter(.config==pull(show_best(fit_lm_mae, metric='mae', n=1),.config)) %>% # only bring back predictions for the best model
    ggplot(aes(x=.pred, y=price )) + 
    geom_point(shape=1)  +
    geom_abline(slope=1, linetype = "dashed", colour='blue') +
    coord_obs_pred() +
    ggtitle("Actuals vs Prediction on Resamples") + 
    theme(text = element_text(size=20))




#> 1. Recette
#> 2. Spécification du modèle
#> 3. Resampling (cv ou bootstrap)
#> 4. Intégration dans un workflow
#> 












