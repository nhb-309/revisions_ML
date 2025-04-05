#source("1_import.R")
donT <- readRDS("donT.RDS")
donT <- as.data.frame(donT)
library(FactoMineR)
indY <- which(names(donT)=="Y")
resACP <- PCA(donT[,-indY])
indm <- grep("_min",colnames(donT))
colnames(donT)[indm]
dim(donT)
donT <- donT[,-indm]
dim(donT)
indm <- grep("_max",colnames(donT))
colnames(donT)[indm]
donT <- donT[,-indm]
dim(donT)
plot(density(donT$Y))
#sur les conseils du métier
donF <- donT[donT$Y>100,]
dim(donF)
indY <- which(names(donT)=="Y")
resACP <- PCA(donF[,-indY])
## redondance sur le vent 
plot(donT$Absolute_wind_direction,donT$Absolute_wind_direction_c)
## redondance sur la nacelle
plot(donT$Nacelle_angle,donT$Nacelle_angle_c)
#donc on vire Absolute_wind_direction et Nacelle_angle
donT$Absolute_wind_direction <- NULL
donT$Nacelle_angle <- NULL
### et dernier pb (avant d'en découvrir d'autres)
### les angles qui varient de 0 (Nord) à 360 degrés (Nord)
### donc il faut passer en coordonnées polaires
donT$sinus_Nacelle_angle <- sin(donT$Nacelle_angle_c/360*2*pi)
donT$cosinus_Nacelle_angle <- cos(donT$Nacelle_angle_c/360*2*pi)
donT$Nacelle_angle_c <- NULL
donT$sinus_Absolute_wind_direction <- sin(donT$Absolute_wind_direction_c/360*2*pi)
donT$cosinus_Absolute_wind_direction <- cos(donT$Absolute_wind_direction_c/360*2*pi)
donT$Absolute_wind_direction_c <- NULL
saveRDS(donF,"donF.RDS")
saveRDS(donF[1:6306,],"donapp.RDS")
saveRDS(donF[-c(1:6306),],"dontest.RDS")
