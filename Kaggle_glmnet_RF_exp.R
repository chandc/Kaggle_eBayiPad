# KAGGLE COMPETITION - GETTING STARTED

# We are adding in the argument stringsAsFactors=FALSE, since we have some text fields
library(caret)
library(glmnet)
library(gbm)
library(randomForest)

M1 <- c("character","numeric","numeric","factor","factor","factor",
        "factor","factor","factor","factor","numeric") 
M2 <- c("character","numeric","numeric","factor","factor","factor",
        "factor","factor","factor","numeric") 
Main = read.csv("eBayiPadTrain.csv", stringsAsFactors=FALSE,colClasses=M1)
eBayTest = read.csv("eBayiPadTest.csv", stringsAsFactors=FALSE,colClasses=M2)

M5 <- rbind(Main[,2:9],eBayTest[,2:9])
MainDummy <- dummyVars("~ biddable+startprice+condition+cellular+carrier+color+
                       storage+productline",data=M5, fullRank=F)
MainDummy <- dummyVars("~ biddable+startprice+condition+color+cellular+carrier+
                       storage+productline+startprice*productline+
                       startprice*condition+startprice*storage+startprice*biddable+startprice*cellular+startprice*carrier",
                       data=M5, fullRank=F)

MainDummy5 <- dummyVars("~ biddable+startprice+condition+color+
                       storage+productline+startprice*storage+startprice*productline+startprice*biddable",data=M5, fullRank=F)

x_all <- as.data.frame(predict(MainDummy,M5))
outcomeName <- 'sold'
predictorsNames <- names(x_all)[names(x_all) != outcomeName]

DF <- x_all[1:nrow(Main),]
Submit <- x_all[(nrow(Main)+1):nrow(x_all),]
DF$sold <- Main$sold

library(caTools)
library(pROC)
set.seed(88)
split = sample.split(DF$sold, SplitRatio = 0.8)
Train = subset(DF, split == TRUE)
Test = subset(DF, split == FALSE)

#
# We will just create a simple logistic regression model, to predict Sold using Price:
# 10 folds cross validation
#
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

#try RandomForest


rf <- randomForest(Train[,predictorsNames], Train[,outcomeName], ntree=200, mtry=10, do.trace=TRUE)

pred_rf <- predict(rf, Test[,predictorsNames],type="prob")
table(Test$sold, pred_rf[,2]>0.5)
library(ROCR)
ROCRpredTest = prediction(pred_rf[,2], Test$sold)
auc = as.numeric(performance(ROCRpredTest, "auc")@y.values)
auc

pred_hybrid <- 0.5*(pred_glmnet+pred_rf[,2])
table(Test$sold, pred_hybrid>0.5)
ROCRpredTest = prediction(pred_hybrid, Test$sold)
auc = as.numeric(performance(ROCRpredTest, "auc")@y.values)
auc


#svm method
# results are very sensitive to the values of gamma
# use default value
library("e1071")

# use svm.tune to optimize parameters

SVM_model <- svm( x = as.matrix(Train[,predictorsNames]), 
                  y = Train[,outcomeName],
                  type="C-classification",
                  kernel="radial",
                  cost=93,gamma = 0.001,
                  probability = TRUE, scale = TRUE, cross=5
)
pred_SVM <- attr( predict(SVM_model, Test[,predictorsNames],probability=TRUE),"probabilities")
ROCRpredTest = prediction(pred_SVM[,2], Test$sold)
auc = as.numeric(performance(ROCRpredTest, "auc")@y.values)
auc

pred_hybrid_2 <- (pred_glmnet+pred_rf[,2]+pred_SVM[,2])/3.0
pred_hybrid_99 <- (pred_glmnet+pred_rf[,2])/2.0
table(Test$sold, pred_hybrid_2>0.5)
ROCRpredTest = prediction(pred_hybrid_2, Test$sold)
auc = as.numeric(performance(ROCRpredTest, "auc")@y.values)
auc

#prepare submission file
pred_1 <- attr( predict(SVM_model, Submit[,predictorsNames],probability=TRUE),"probabilities")
pred_2 <- predict(rf, Submit[,predictorsNames],type="prob")
pred_3 <- predict(model, as.matrix(Submit[,predictorsNames]), type="response",s = "lambda.min")

Pred <- (pred_1[,2]+pred_2[,2]+pred_3)/3.0


MySubmission = data.frame(UniqueID = eBayTest$UniqueID, Probability1 = Pred[,1])

write.csv(MySubmission, "SubmissionHybrid.csv", row.names=FALSE)



