# Chargement des librairies
library(tidyverse)
library(rpart)
library(pROC)
 
# Import des données
db=read.csv('SAheart.csv',
            stringsAsFactors = T)%>% 
    mutate(famhist = case_when(famhist=='Present'~1,
                               famhist=='Absent'~0)) %>% 
    mutate(famhist=as.factor(famhist)) %>% 
    mutate(Y = case_when(chd=='No'~0,
                         chd=='Si'~1),
           Y=as.factor(Y)) %>% 
    select(-chd)

# Apprentissage et test
app_index=sample(nrow(db),0.75*nrow(db))
app=db[app_index,]
test=db[-app_index,]

# L'objectif est de maximiser la pureté de chaque feuille. 
# Pour y arriver on pourrait paramétrer un arbre très profond. 
# Mais on risque le sur apprentissage. 
## Comment faire ? 
# Construire un arbre maximal qu'on vient ensuite élaguer. 


### On commence par construire un arbre très profond    
arbre=rpart(Y~.,data=app,cp=0.00000000002)

#printcp(arbre) 
    # CP = complexité ; 
    # rel error = erreur sur jeu d'entraînement
    # xerror = erreur par VC

### Paramètre de pruning tel qu'on minimise l'erreur par VC
cp_opt=arbre$cptable %>% 
    as.data.frame() %>% 
    filter(xerror==min(xerror)) %>% 
    select(CP) %>% max() %>% as.numeric()  
            
# on part de l'arbre profond pour construire l'arbre élagué.                              
arbre.fin = prune(arbre,cp=cp_opt)

# DF qui oppose aux Y observés, les prédictions de l'arbre profond 
# et de l'arbre élagué
summ=data.frame(large=predict(arbre,newdata=test,type='class'),
                fin=predict(arbre.fin,newdata=test,type='class'),
                Y=test$Y)

# On compare les taux de mauvaise classification
table=summ %>% 
    summarise_all(~mean(Y!=.)) %>% 
    select(-Y)

table

# Courbes ROC
score = data.frame(
    large=predict(arbre,newdata=test)[,2],
    fin = predict(arbre.fin,newdata=test)[,2],
    obs=test$Y)


plot(roc(score$obs,score$large),col='darkred',lty='dotted')
lines(roc(score$obs,score$fin),col='black')


# Variables d'importance dans l'arbre élagué
var.imp=arbre.fin$variable.importance
    var.imp # valeur d'importance de chaque variable explicative dans l'arbre
nom.var=substr(names(var.imp),1,3)
    nom.var # purement esthétique = lisibilité +++ 
nom.var[c(4,5)]=c("co.c","co.p")
var.imp1=data.frame(var=nom.var,score=var.imp) # var.imp et nom.var dans le même data.frame
    var.imp1
var.imp1$var=factor(var.imp1$var, levels=nom.var) # mise en facteur

ggplot(var.imp1)+aes(x=var,y=score)+geom_bar(stat='identity')+ 
    theme_classic() # graphique
