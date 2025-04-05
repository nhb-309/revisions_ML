#source("2_nettoyage.R")
rm(list=ls())
don <- readRDS("donapp.RDS")
library(glmnet)
library(rpart)
library(randomForest)
library(gbm)
donX <- model.matrix(Y~.,data=don)
indY <- which(names(don)=="Y")
donY <- don[,indY]
nbloc=10
set.seed(1234)
bloc <- sample(rep(1:nbloc,length=nrow(don)))
## un df de res
RES <- data.frame(Y=don$Y,mco=NA,step=NA,
                  ridge=NA,lasso=NA,elast=NA,
                  arbre=NA,foret=NA,gbm=NA) 
for(ii in 1:nbloc){
  print(ii)
  donA <- don[bloc!=ii,]
  donT <- don[bloc==ii,]
  donXA <- donX[bloc!=ii,]
  donXT <- donX[bloc==ii,]
  donYA <- donY[bloc!=ii]
  ###mco
  tmp <- lm(Y~.,data=donA)
  RES[bloc==ii,"mco"] <- predict(tmp,donT)
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
