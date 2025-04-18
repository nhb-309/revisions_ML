library(glmnet)
library(bestglm)
library(ranger)
library(pROC)
library(ggplot2)
library(tidyverse)


data(SAheart)

maladie = SAheart
maladie$chd = as.factor(maladie$chd)

col_num = sapply(maladie, is.numeric)
col_fct = sapply(maladie, is.factor)
scaled_maladie = sapply(maladie[,col_num],scale)

maladie=cbind(scaled_maladie, maladie[col_fct])
maladie$famhist=ifelse(maladie$famhist == 'Present', 1,0)
maladie %>% head()

SCORE = data.frame(Y=maladie$chd)
SCORE

nbbloc=5
bloc = rep(0, nrow(maladie))

ind0=which(maladie$chd == 0)
ind1=which(maladie$chd == 1)

bloc[ind0] = sample(rep(1:nbbloc,length = length(ind0)))
bloc[ind1] = sample(rep(1:nbbloc,length = length(ind1)))

maladieX = model.matrix(chd~. , data=maladie)[,-1]
maladieY = maladie[,"chd"]


models = list()

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

    maladieXA = maladieX[bloc!=i,]
    maladieYA = maladieY[bloc!=i]
    maladieXT = maladieX[bloc==i,]

    #ridge
    tmp = cv.glmnet(maladieXA, maladieYA, alpha=0, family='binomial')
    SCORE[bloc==i,'ridge'] = predict(tmp,maladieXT,"lambda.1se",type='response')
    
    #lasso
    tmp = cv.glmnet(maladieXA, maladieYA, alpha=1, family='binomial')
    SCORE[bloc==i,'lasso'] = predict(tmp,maladieXT,"lambda.1se",type='response')
    
    #elnet
    tmp = cv.glmnet(maladieXA, maladieYA, alpha=0.5, family='binomial')
    SCORE[bloc==i,'elnet'] = predict(tmp,maladieXT,"lambda.1se",type='response')
    
    
    # logistique 
    #mod2 <- step(reglog,trace=0)
    #SCORE[bloc==i,"AIC"] <- predict(mod2,maladieT,type="response")
    #mod3 <- step(reglog,trace=0,k=log(nrow(maladieA)))
    #SCORE[bloc==i,"BIC"] <- predict(mod3,maladieT,type="response")
    
    #### foret
    #
    #
    #hpForet = expand.grid(num.trees = c(100,400,600,800),
    #                      max.depth = c(1,3,5),
    #                      min.bucket = c(1,2,3),
    #                      mtry=c(1,2,3,4,5)
    #                      )
    #hpForetParams = data.frame(hpForet,'auc'=NA);hpForetParams
    #
    #for(xx in 1:nrow(hpForet)){
    #    foret <- ranger(factor(chd)~.,data=maladieA,
    #                    classification = T,
    #                    probability = T,
    #                    num.trees=hpForet$num.trees[xx],
    #                    max.depth =hpForet$max.depth[xx])
    #    hpForetResult = predict(foret,maladieT,type='response')
    #    hpForetParams[xx,'auc']=roc(maladieT$chd,hpForetResult$predictions[,2])$auc
    #    
    #}
    #
    #bestForet=hpForetParams[which.max(hpForetParams$auc),] # sélection du meilleur hyper paramétrage
    #
    #bestFitForet=ranger(factor(chd)~.,data=maladieA,
    #               classification = T, probability = T, 
    #               num.trees=bestForet$num.trees,
    #               max.depth = bestForet$max.depth,
    #               min.bucket = bestForet$min.bucket,
    #               mtry = bestForet$mtry)
    #SCORE[bloc==i,"foret"] <- predict(bestFitForet,maladieT,type="response")$predictions[,2]
    
    
}

SCORE


rocCV= roc(Y~., data=SCORE)

aucmodele=sort(round(unlist(lapply(rocCV,auc)),5),dec=TRUE)
aucmodele
lapply(rocCV,FUN=auc) %>% unlist() %>% round(4) 

plot(rocCV$glm,col='blue',lwd=0.5)
lines(rocCV$ridge,col='red',lwd=0.5)
lines(rocCV$elnet,col='black',lwd=0.5)



