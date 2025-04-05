#source("2_nettoyage.R")
path = 'F:/DataScientist/revisions_ML/codes/revisions_ML'; setwd(path)
list.files()
don <- readRDS("donapp.RDS")
library(glmnet)
library(rpart)
library(randomForest)
library(gbm)

# Pour certains algorithmes: il faut utiliser les données sous une forme matricielle.
# Transformation : 1) Matrice et 2) Séparer la target des prédicteurs. 
# cv.glmnet(x= ... , y= ... , type.measure = '...')
# c'est le cas des régressions pénalisées. 

donX <- model.matrix(Y~.,data=don)  # matrice d'apprentissage 
indY <- which(names(don)=="Y")      # indice qui localise la variable target dans les données
donY <- don[,indY]                  # vecteur contenant la variable target

# 10 blocs de validation croisée
nbloc=10
set.seed(1234)

bloc <- sample(rep(1:nbloc,length=nrow(don)))  # création des blocs de la VC

## Un dataframe de résultats
RES <- data.frame(Y=don$Y,mco=NA,step=NA,
                  ridge=NA,lasso=NA,elast=NA,
                  arbre=NA,foret=NA,gbm=NA)    # dataframe de stockage des résultats

for(ii in 1:nbloc){
  print(ii)
  donA <- don[bloc!=ii,]   # dataframe : Jeu d'entraînement
  donT <- don[bloc==ii,]   # dataframe : Jeu de test
  
  # 1) Il faut créer une matrice pour certains algorithmes. 
  # 2) Ces matrices contiennent les prédicteurs d'une part et la target d'autre part
  donXA <- donX[bloc!=ii,] # matrice :   Jeu d'entraînement.
  donXT <- donX[bloc==ii,] # matrice :   Jeu de test.  
  donYA <- donY[bloc!=ii]  # matrice :  
  
  ###mco
  tmp <- lm(Y~.,data=donA)                  # entraînement sur donA.
  RES[bloc==ii,"mco"] <- predict(tmp,donT)  # predict sur dontT. 
  
  ###step
  choix <- step(tmp,trace=0)                  
  RES[bloc==ii,"step"] <- predict(choix,donT) 
  
  ##ridge
  tmp <- cv.glmnet(donXA,donYA,alpha=0)
  RES[bloc==ii,"ridge"] <- predict(tmp,donXT)
  
  ##lasso
  tmp <- cv.glmnet(donXA,donYA,alpha=1)
  RES[bloc==ii,"lasso"] <- predict(tmp,donXT)

  ##ridge
  tmp <- cv.glmnet(donXA,donYA,alpha=0.5)
  RES[bloc==ii,"elast"] <- predict(tmp,donXT)

  ##arbre 
  tmp <- rpart(Y~.,data=donA)
  RES[bloc==ii,"arbre"] <- predict(tmp,donT)
  
  ##foret
  tmp <- randomForest(Y~.,data=donA,ntree=100)
  RES[bloc==ii,"foret"] <- predict(tmp,donT)
  
  ##foret
  tmp <- randomForest(Y~.,data=donA,ntree=100,mtry=floor(sqrt(ncol(donA))))
  RES[bloc==ii,"foret2"] <- predict(tmp,donT)
  ##gbm
  tmp <- gbm(Y~.,data=donA,distribution = "gaussian",cv.folds=5,n.trees=1500,
             interaction.depth = 3,shrinkage = 0.3)
  best.iter <- gbm.perf(tmp, method = "cv")
  RES[bloc==ii,"gbm"] <- predict(tmp,donT,n.trees = best.iter)
}
 RES[RES<0] <- 0
erreur2 <- function(X,Y){mean((X-Y)^2)}
sort(round(apply(RES,2,erreur2,Y=RES[,"Y"])[-1]))
