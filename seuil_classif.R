library(glmnet)
library(bestglm)
library(ranger)

data(SAheart)

maladie = SAheart
maladie$chd = as.factor(maladie$chd)



SCORE = data.frame(Y=maladie$chd)
SCORE

nbbloc=5
bloc = rep(0, nrow(maladie))
ind0=which(maladie$chd == 0)
ind1=which(maladie$chd == 1)

bloc[ind0] = sample(rep(1:nbbloc,length = length(ind0)))
bloc[ind1] = sample(rep(1:nbbloc,length = length(ind1)))

models = list()
i=2
for(i in 1:nbbloc ){
    print(i)
    maladieA = maladie[bloc!=i,]
    maladieT = maladie[bloc == i,]
    
    mat_maladieX = model.matrix(chd~., data=maladie)
    mat_maladieY = maladie$chd
    
    # logistique global    
    reglog = glm(chd~., data=maladieA, family = 'binomial')
    SCORE[bloc == i, 'glm']= predict(reglog, maladieT, type = 'response')
    
    choix = bestglm(maladieA, family=binomial)
    SCORE[bloc == i, "choix"] = predict(choix$BestModel, maladieT, type='response')
    
    tmp = choix$BestModels[1,-ncol(maladie)]
    var = names(tmp)[tmp==TRUE]
    models[[i]]=var
    
    # logistique 
    mod2 <- step(reglog,trace=0)
    SCORE[bloc==i,"AIC"] <- predict(mod2,maladieT,type="response")
    mod3 <- step(reglog,trace=0,k=log(nrow(maladieA)))
    SCORE[bloc==i,"BIC"] <- predict(mod3,maladieT,type="response")
    
    ### foret
    hpForet = expand.grid(num.trees = c(100,400,600,800),
                          max.depth = c(1,3,5),
                          min.bucket = c(1,2,3),
                          mtry=c(1,2,3,4,5)
                          )
    hpForetParams = data.frame(hpForet,'auc'=NA);hpForetParams
    for(xx in 1:nrow(hpForet)){
        foret <- ranger(factor(chd)~.,data=maladieA,
                        classification = T,
                        probability = T,
                        num.trees=hpForet$num.trees[xx],
                        max.depth =hpForet$max.depth[xx])
        hpForetResult = predict(foret,maladieT,type='response')
        hpForetParams[xx,'auc']=roc(maladieT$chd,hpForetResult$predictions[,2])$auc
        
    }
    bestForet=hpForetParams[which.max(hpForetParams$auc),]
    bestFitForet=ranger(factor(chd)~.,data=maladieA,
                   classification = T, probability = T, 
                   num.trees=bestForet$num.trees,
                   max.depth = bestForet$max.depth,
                   min.bucket = bestForet$min.bucket,
                   mtry = bestForet$mtry)
    SCORE[bloc==i,"foret"] <- predict(foret,maladieT,type="response")$predictions[,2]
    
    
}

SCORE

###### Résultats hyperparamétrage ----
## 1. Exploration de l'hyperparamétrage Foret ----
ggplot(hpForetParams)+
    aes(x=num.trees,y=auc,group=num.trees,col=as.factor(max.depth))+
    geom_jitter(alpha=0.7)+theme_light()



SCORE


library(pROC)

rocCV= roc(Y~., data=SCORE)
SCORE
aucmodele=sort(round(unlist(lapply(rocCV,auc)),3),dec=TRUE)

lapply(rocCV,FUN=auc) %>% unlist()
#auc(rocCV$glm) 
#auc(rocCV$choix)
#auc(rocCV$foret)


ind=order(aucmodele,decreasing=T); L = length(ind)
mapply(plot, rocCV[ind[1:L]],col = 1:L,add=T)
legend("bottomright",legend = names(SCORE)[2:L])
mapply(plot, rocCV[ind[1:L]],col = 1:L, lty=1:L, legacy.axes = T, lwd=1, add = c(T))
mapply(plot, rocCV[ind[1:L]],col = 1:L, lty=1:L, legacy.axes = T, lwd=1, add = c(T,T,T,T,T,T))
legend("bottomright",legend = names(SCORE)[2:L], col=2:L, lty=2:L, lwd=1,cex=1)


tmp = lapply(rocCV, FUN = coords, x='best', ret = c('threshold','tp','fp','tn','fn','sensitivity','specificity','ac'),transpose = T)

mat=do.call(rbind, tmp)
mat
