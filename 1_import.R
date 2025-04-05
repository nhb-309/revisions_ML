library(data.table)
X <- fread("engieX.csv")
Y <- fread("engieY.csv")
don <- merge(X,Y)
dim(don)
colnames(don)[ncol(don)] <- "Y"
saveRDS(don,"don.RDS")
###############################################
don <- readRDS("don.RDS")
don <- don[,-1]
don <- don[don$MAC_CODE=="WT1",]
don <- don[,-1]
plot(don$Grid_voltage)
plot(don$Date_time,don$Grid_voltage)
tmp <-apply(is.na(don),2,sum)
tmp[tmp>0.1*nrow(don)]
library(tidyverse)
don <- select_if(don,apply(is.na(don),2,sum)<0.1*nrow(don))
dim(don)
plot(diff(don$Date_time))
#petit pb les durées ne se suivent pas
#mais on s'en fout !
don2 <- na.omit(don)
dim(don2)
don2$Date_time[1:10]
sel <- don2$Date_time%%6
table(sel)
donT <- don2[sel==2,]
donT$Date_time <- NULL
saveRDS(donT,"donT.RDS")
 