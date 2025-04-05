library(glmnet)
library(bestglm)

data(SAheart)

maladie = SAheart
maladie$chd = as.factor(maladie$chd)
dim(maladie)
table(maladie$chd)


SCORE = data.frame(Y=maladie$chd)
SCORE

nbbloc=5
bloc = rep(0, nrow(maladie))
ind0=which(maladie$chd == 0)
ind1=which(maladie$chd == 1)

bloc[ind0] = sample(rep(1:nbbloc,length = length(ind0)))
bloc[ind1] = sample(rep(1:nbbloc,length = length(ind1)))

models = list()

for(i in 1:nbbloc ){
    print(i)
    maladieA = maladie[bloc!=i,]
    maladieT = maladie[bloc == i,]
    # logistique global    
    reglog = glm(chd~., data=maladieA, family = 'binomial')
    SCORE[bloc == i, 'glm']= predict(reglog, maladieT, type = 'response')
    
    choix = bestglm(maladieA, family=binomial)
    SCORE[bloc == i, "choix"] = predict(choix$BestModel, maladieT, type='response')
    
    tmp = choix$BestModels[1,-ncol(maladie)]
    var = names(tmp)[tmp==TRUE]
    models[[i]]=var
}


SCORE


library(pROC)

rocCV= roc(Y~., data=SCORE)

aucmodele=sort(round(unlist(lapply(rocCV,auc)),3),dec=TRUE)

lapply(rocCV,FUN=auc)
auc(rocCV$glm) 
auc(rocCV$choix)

ind=order(aucmodele,decreasing=T)
mapply(plot, rocCV[ind[1:2]],col = 1:2, lty=1:2, legacy.axes = T, lwd=3, add = c(F,T))
legend("bottomright",legend = names(SCORE)[2:3], col=1:2, lty=1:2, lwd=3,cex=1)

tmp = lapply(rocCV, FUN = coords, x='best', ret = c('threshold','tp','fp','tn','fn','sensitivity','specificity','ac'),transpose = T)

mat=do.call(rbind, tmp)
mat
