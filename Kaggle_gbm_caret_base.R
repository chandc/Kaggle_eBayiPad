library(caTools)
library(pROC)
library(caret)

M1 <- c("character","numeric","numeric","factor","factor","factor",
        "factor","factor","factor","factor","numeric") 
M2 <- c("character","numeric","numeric","factor","factor","factor",
        "factor","factor","factor","numeric") 
Main = read.csv("eBayiPadTrain.csv", stringsAsFactors=FALSE,colClasses=M1)
eBayTest = read.csv("eBayiPadTest.csv", stringsAsFactors=FALSE,colClasses=M2)

MainDummy <- dummyVars("~ biddable+startprice+condition+cellular+carrier+color+
                       storage+productline",data=Main, fullRank=F)

MainDummy2 <- dummyVars("~ biddable+startprice+condition+cellular+carrier+color+
                       storage+productline+biddable*startprice+startprice*productline+startprice*storage",
                       data=Main, fullRank=F)
MainDummy3 <- dummyVars("~ biddable+startprice+condition+cellular+carrier+color+
                       storage*startprice+productline*startprice",
                       data=Main, fullRank=F)

MainDummy <- dummyVars("~ biddable+startprice+condition+color+cellular+carrier+
                       storage+productline+startprice*productline+
                       startprice*condition+startprice*storage+startprice*biddable+startprice*cellular+startprice*carrier",
                       data=M5, fullRank=F)

DF <- as.data.frame(predict(MainDummy,Main))
DF$sold <- Main$sold
# save the outcome for the glmnet model
tempOutcome <- DF$sold  

DF$sold <- ifelse(DF$sold==1,'yes','no')
print(names(DF))


set.seed(88)
split = sample.split(DF$sold, SplitRatio = 0.8)
Train = subset(DF, split == TRUE)
Test = subset(DF, split == FALSE)

outcomeName <- 'sold'
predictorsNames <- names(DF)[names(DF) != outcomeName]

# run model
gbmControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           summaryFunction = twoClassSummary, classProbs = TRUE)

gbmModel <- train(Train[,predictorsNames], as.factor(Train[,outcomeName]), 
                  method='gbm', 
                  trControl=gbmControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"))

summary(gbmModel)
print(gbmModel)

#################################################
# evalutate model
#################################################
# get predictions on your testing data

# class prediction
predictions <- predict(object=gbmModel, Test[,predictorsNames], type='raw')
head(predictions)
postResample(pred=predictions, obs=as.factor(Test[,outcomeName]))
# probabilities 
predictions <- predict(object=gbmModel, Test[,predictorsNames], type='prob')
head(predictions)

postResample(pred=predictions[[2]], obs=ifelse(Test[,outcomeName]=='yes',1,0))
auc <- roc(ifelse(Test[,outcomeName]=="yes",1,0), predictions[[2]])
print(auc$auc)

################################################
# advanced stuff
################################################

# boosted tree model (gbm) adjust learning rate and and trees
gbmGrid <-  expand.grid(n.trees = c(100,125,150),
                        interaction.depth =  c(5,7),
                        shrinkage = c(0.01,0.1),
                        n.minobsinnode = c(10,20))

# run model

gbm2Model <- train(Train[,predictorsNames], as.factor(Train[,outcomeName]), 
                  method='gbm', 
                  trControl=gbmControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"),
                  tuneGrid = gbmGrid, verbose=F)


# get predictions on your testing data
# probabilities 
predictions <- predict(object=gbm2Model, Test[,predictorsNames], type='prob')
pred_gbm2 <- predictions
head(predictions)

postResample(pred=predictions[[2]], obs=ifelse(Test[,outcomeName]=='yes',1,0))
auc <- roc(ifelse(Test[,outcomeName]=="yes",1,0), predictions[[2]])
print(auc$auc)


# display variable importance on a +/- scale 
vimp <- varImp(gbmModel, scale=F)
results <- data.frame(row.names(vimp$importance),vimp$importance$Overall)
results$VariableName <- rownames(vimp)
colnames(results) <- c('VariableName','Weight')
results <- results[order(results$Weight),]
results <- results[(results$Weight != 0),]

par(mar=c(5,15,4,2)) # increase y-axis margin. 
xx <- barplot(results$Weight, width = 0.85, 
              main = paste("Variable Importance -",outcomeName), horiz = T, 
              xlab = "< (-) importance >  < neutral >  < importance (+) >", axes = FALSE, 
              col = ifelse((results$Weight > 0), 'blue', 'red')) 
axis(2, at=xx, labels=results$VariableName, tick=FALSE, las=2, line=-0.3, cex.axis=0.6)  



# We will just create a simple logistic regression model, to predict Sold using Price:
model = cv.glmnet(
  x = as.matrix(Train[,predictorsNames]), 
  y = Train[,outcomeName], 
  family = "binomial",
  standardize = T,
  alpha = 0.5,
  type.measure = "auc",
  intercept = T,
  nfolds = 10)

cat("Max cv.glmnet VAL AUC:", max(model$cvm, na.rm = T), "\n")

# And then make predictions on the test set:
pred_glmnet = predict(model, as.matrix(Test[,predictorsNames]), type="response",s = "lambda.1se")
library(ROCR)
ROCRpredTest = prediction(pred_glmnet, Test$sold)
auc = as.numeric(performance(ROCRpredTest, "auc")@y.values)
auc

pred_hybrid_2 <- (pred_glmnet+pred_gbm2[,2])/2.0
table(Test$sold, pred_hybrid_2>0.5)
ROCRpredTest = prediction(pred_hybrid_2, Test$sold)
auc = as.numeric(performance(ROCRpredTest, "auc")@y.values)
auc



