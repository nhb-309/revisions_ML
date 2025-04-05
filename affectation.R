###affectation
donT <- readRDS("dontest.RDS")
donA1 <- readRDS("don1.RDS")
centre1 <- colMeans(donA1[,-ncol(donA1)])
donA2 <- readRDS("don2.RDS")
centre2 <- colMeans(donA2[,-ncol(donA2)])
donA3 <- readRDS("don3.RDS")
centre3 <- colMeans(donA3[,-ncol(donA3)])
centre <- rbind(centre1,centre2,centre3)
distance <- function(X,centre){mean((X-centre)^2)}
##matrice distance des indi aux 3 centres
MAT <- matrix(NA,nrow=nrow(donT),ncol=3)
for(ii in 1:nrow(donT)){
  for(jj in 1:3){
    MAT[ii,jj] <- distance(unlist(donT[ii,-ncol(donT)]),centre[jj,])
  }
}
gpT <- apply(MAT,1,which.min)
table(gpT)
saveRDS(gpT,"gptest.RDS")
