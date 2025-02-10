library(ggplot2)
library(yardstick)
library(workflows)
library(parsnip) 
library(tune)
library(recipes)
library(tidymodels)
library(readr)
library(broom.mixed)
library(dotwhisker)
library(tidyverse) #core tidyverse
library(tidymodels) # tidymodels framework
library(glmnet) # elastic net logistic regression
library(themis) # provides up/down-sampling methods for the data
library(readr)       # for importing data
library(vip)  


# Process des données = Train/Validation + Test
# Modèle 1 - Régression Logistique pénalisée
  # Recipe 
  # Workflow 
# Modèle 2 - Tree bagging
  # Recipe
  # Workflow
  


hotels =
  read_csv("https://tidymodels.org/start/case-study/hotels.csv") %>%
  mutate(across(where(is.character), as.factor))

glimpse(hotels)

hotels %>% 
  count(children) %>% 
  mutate(prop=n/sum(n))

################################################################################
splits=initial_split(hotels, strata=children) 

# Echantillon de l'échantillon d'entraînement et de test
hotels_other=training(splits) # Other contient training et validation
hotels_test=testing(splits)   # Uniquement testing

# On a stratifié donc les échantillons de tr/va et te sont équilibrés. 
hotels_other %>% 
  count(children) %>% 
  mutate(prop=n/sum(n))

hotels_test %>% 
  count(children) %>% 
  mutate(prop=n/sum(n))

# échantillon de validation
set.seed(234)
val_set = validation_split(hotels_other, 
                            strata = children, 
                            prop = 0.80)

val_set
###############################################################################

# Régression logistique

lr_mod = logistic_reg(penalty=tune(), # tune() pour optimiser l'hyperparamètre
                      mixture=1) %>%  # Mixture = paramètre de régularisation L1/L2, elasticnet  
  set_engine('glmnet')
  
holidays = c("AllSouls", "AshWednesday", "ChristmasEve", "Easter", 
              "ChristmasDay", "GoodFriday", "NewYearsDay", "PalmSunday")

lr_recipe =
  recipe(children ~ ., data = hotels_other) %>%  # 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date, holidays = holidays) %>% 
  step_rm(arrival_date) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

lr_recipe

# le workflow combine le modèle et la transformation qu'on opère sur les données.
lr_workflow=
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(lr_recipe)

# On définit la grille d'hyperparamétrage
# length.out = 30 // cela signifie qu'on va tester 30 valeurs d'hyperparamètre 
lr_reg_grid=tibble(penalty=10^seq(-4,-1, length.out=30))

# stockage des résultats de l'apprentissage
lr_res =
  lr_workflow %>% # lr_workflow combine la recipe et le modèle qu'on a défini. 
  tune_grid(val_set, # On utilise le jeu de validation ici. 
            grid = lr_reg_grid, # on a défini notre grille juste au dessus. 
            control = control_grid(save_pred = TRUE), # stocke l'entraînement. 
            metrics = metric_set(roc_auc)) # 

lr_plot = 
  lr_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

lr_plot 

top_models=
  lr_res %>% 
  show_best(metric = "roc_auc", n = 15) %>% 
  arrange(penalty) 
top_models

lr_best =
  lr_res %>% 
  collect_metrics() %>% 
  arrange(penalty) %>% 
  slice(12)
lr_best

lr_auc =
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(children, .pred_children) %>% 
  mutate(model = "Logistic Regression")

autoplot(lr_auc)

################################################################################
cores=parallel::detectCores()
rf_mod=
  rand_forest(mtry=tune(), min_n=tune(),trees=1000) %>% 
  set_engine('ranger',num.threads=cores) %>% 
  set_mode('classification')

rf_recipe= 
  recipe(children ~ ., data = hotels_other) %>% 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date) %>% 
  step_rm(arrival_date) 
rf_workflow= 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(rf_recipe)

rf_mod

#> Random Forest Model Specification (classification)
#> 
#> Main Arguments:
#>   mtry = tune()
#>   trees = 1000
#>   min_n = tune()
#> 
#> Engine-Specific Arguments:
#>   num.threads = cores
#> 
#> Computational engine: ranger

# show what will be tuned

extract_parameter_set_dials(rf_mod)

#> Collection of 2 parameters for tuning
#> 
#>  identifier  type    object
#>        mtry  mtry nparam[?]
#>       min_n min_n nparam[+]
#> 
#> Model parameters needing finalization:
#>    # Randomly Selected Predictors ('mtry')
#> 
#> See `?dials::finalize` or `?dials::update.parameters` for more information.

c1=Sys.time()
set.seed(345)
rf_res <- 
  rf_workflow %>% 
  tune_grid(val_set,
            grid = 25,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))
c2=Sys.time()
c2-c1


