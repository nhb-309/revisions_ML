############################
don <- readRDS("donapp.RDS")
donT <- readRDS("dontest.RDS")
dim(don)
#### je compare les algos, je garde le meilleur et
#### je prévois les 10 000 individus restants
#### j'estime avec la foret et les mêmes paramètres
#### sur les 6306 et je prévois les autres
tmp <- randomForest(Y~.,data=don,ntree=100)
prevF <- predict(tmp,donT)
mean((donT$Y-prevF)^2)
##### maintenant les groupes
inertie <- 1:10
for(ii in 1:10){
  tmp = kmeans(don[,-38],centers=ii,nstart = 10)
  inertie[ii] = tmp$betweenss/tmp$totss*100
}
plot(inertie,type="h")
# 4 ou 5 :(((( groupes
doncr <- scale(don[,-38])
inertie <- 1:10
for(ii in 1:10){
  tmp = kmeans(doncr,centers=ii,nstart = 10)
  inertie[ii] = tmp$betweenss/tmp$totss*100
}
plot(inertie,type="h")
#### donc pas CR
tmp = kmeans(don[,-38],centers=4,nstart = 10)
gp=tmp$cluster
boxplot(don$Y~gp)
tmp = kmeans(don[,-38],centers=3,nstart = 10)
gp=tmp$cluster
boxplot(don$Y~gp)
#### 3 groupes cela me va bien
don1 = don[gp==1,]
don2 = don[gp==2,]
don3 = don[gp==3,]
saveRDS(don1,"don1.RDS")
don1 <- readRDS("don1.RDS")
saveRDS(don2,"don2.RDS")
don2 <- readRDS("don2.RDS")
saveRDS(don3,"don3.RDS")
don3 <- readRDS("don3.RDS")
####
gpT <- readRDS("gptest.RDS")

PREVfin = data.frame(Y=donT$Y,prev1gp=NA,prev3gp=NA)
tmp <- randomForest(Y~.,data=don,ntree=100)
prevF <- predict(tmp,donT)
mean((donT$Y-prevF)^2)
PREVfin$prev1gp=prevF
####GP1, step qui gagne
mod1 <- step(lm(Y~.,data=don1),trace=0)
prev1 <- predict(mod1,donT[gpT==1,])
PREVfin[gpT==1,"prev3gp"]=prev1
plot(prev1,donT[gpT==1,"Y"])
mean((prev1-donT[gpT==1,"Y"])^2)
####GP2
mod2 <- randomForest(Y~.,data=don2,ntree=100)
prev2 <- predict(mod2,donT[gpT==2,])
PREVfin[gpT==2,"prev3gp"]=prev2
plot(prev2,donT[gpT==2,"Y"])
mean((prev2-donT[gpT==2,"Y"])^2)
####GP3
mod3 <- randomForest(Y~.,data=don3,ntree=100,mtry=floor(sqrt(ncol(don3))))
prev3 <- predict(mod3,donT[gpT==3,])
PREVfin[gpT==3,"prev3gp"]=prev3
plot(prev3,donT[gpT==3,"Y"])
mean((prev3-donT[gpT==3,"Y"])^2)

prev <- c(prev1,prev2,prev3)
YY <- c(donT[gpT==1,"Y"],donT[gpT==2,"Y"],donT[gpT==3,"Y"])
mean((prev-YY)^2)

apply(PREVfin,2,erreur2,Y=PREVfin$Y)

erreurfin <- data.frame(groupe=gpT,ergl=(PREVfin$Y-PREVfin$prev1gp)^2,
                    erg3g=(PREVfin$Y-PREVfin$prev3gp)^2)
library(tidyverse)
erreurfin %>% group_by(groupe) %>%summarise_all(mean) 
