#Clear the environment 

rm(list=ls(all=TRUE))

#Read the input data that is given

setwd("D:/INSOFE/CSE7305c/CUTe 3")

company_data<-read.csv("train.csv",header = T)

test_data<-read.csv("test.csv",header=T)

#Use head() and tail() functions to get a feel of the data

head(company_data)

tail(company_data)

head(test_data)

tail(test_data)

#Check the structure of the input data

str(company_data)

str(test_data)

#Check the distribution of the input data using the summary function

summary(company_data)

summary(test_data)

sum(is.na(company_data))

sum(is.na(test_data))

#Removing columns that are not necessary
company_data_mod<-company_data[,-c(1,66)]

test_data_mod<-test_data[-c(1)]

#Impute and standardize the data 

company_data_mod[is.na(company_data_mod)]<-0

test_data_mod[is.na(test_data_mod)]<-0

#After imputation, add target colum back to the train data
company_data_mod<-cbind(company_data_mod,target=company_data$target)

company_data_mod$target<-as.factor(as.character(company_data_mod$target))

library(caret)

# Divide the data into test and validation
set.seed(123)

train_RowIDs = createDataPartition(company_data_mod$target,p=0.8,list=F)
train = company_data_mod[train_RowIDs,]
validation= company_data_mod[-train_RowIDs,]
rm(train_RowIDs)

#Install Packages
install.packages("randomForest")
library(randomForest)
# Model Building -

set.seed(123)

# Build the classification model using randomForest
model = randomForest(target ~ ., data=train, 
                     keep.forest=TRUE, ntree=100) 

# Print and understand the model
print(model)

# Important attributes
model$importance  
round(importance(model), 2) 


# Extract and store important variables obtained from the random forest model
rf_Imp_Attr = data.frame(model$importance)
rf_Imp_Attr = data.frame(row.names(rf_Imp_Attr),rf_Imp_Attr[,1])
colnames(rf_Imp_Attr) = c('Attributes', 'Importance')
rf_Imp_Attr = rf_Imp_Attr[order(rf_Imp_Attr$Importance, decreasing = TRUE),]


# plot (directly prints the important attributes) 
varImpPlot(model)

#Predict on Train data 
pred_Train = predict(model, 
                     train[,setdiff(names(train), "target")],
                     type="response", 
                     norm.votes=TRUE)

# Build confusion matrix and find accuracy   
cm_Train = table("actual"= train$target, "predicted" = pred_Train);
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
rm(pred_Train, cm_Train)

# Predicton Validation Data
pred_Validation = predict(model, validation[,setdiff(names(validation),
                                              "target")],
                    type="response", 
                    norm.votes=TRUE)

# Build confusion matrix and find accuracy   
cm_Validation = table("actual"=validation$target, "predicted"=pred_Validation);
accu_Validation= sum(diag(cm_Validation))/sum(cm_Validation)
rm(pred_Validation, cm_Validation)

accu_Train
accu_Validation
rf_Imp_Attr$Attributes

# Build randorm forest using all attributes. 
top_Imp_Attr = as.character(rf_Imp_Attr$Attributes[1:64])

set.seed(15)

# Build the classification model using randomForest
model_Imp = randomForest(target~.,
                         data=train[,c(top_Imp_Attr,"target")], 
                         keep.forest=TRUE,ntree=100) 

# Print and understand the model
print(model_Imp)

# Important attributes
model_Imp$importance  

# Predict on Train data 
pred_Train = predict(model_Imp, train[,top_Imp_Attr],
                     type="response", norm.votes=TRUE)


# Predicton Test Data
pred_Validation= predict(model_Imp, validation[,top_Imp_Attr],
                    type="response", norm.votes=TRUE)
table(pred_Validation)

library(caret)

result_Train<-confusionMatrix(pred_Train, train$target)

result_Train

result_Train$byClass[7]

result_Validation<-confusionMatrix(pred_Validation, validation$target)

result_Validation

result_Validation$byClass[7]

#Select mtry value with minimum out of bag(OOB) error.

mtry <- tuneRF(train[,-65],train$target, ntreeTry=100,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

set.seed(71)
rf <- randomForest(target~.,data=train, mtry=best.m, importance=TRUE,ntree=100)
print(rf)

#Evaluate variable importance
importance(rf)

# Important attributes
rf$importance  
round(importance(rf), 2)   

# Extract and store important variables obtained from the random forest model
rf_Imp_Attr = data.frame(rf$importance)
rf_Imp_Attr = data.frame(row.names(rf_Imp_Attr),rf_Imp_Attr[,1])
colnames(rf_Imp_Attr) = c('Attributes', 'Importance')
rf_Imp_Attr = rf_Imp_Attr[order(rf_Imp_Attr$Importance, decreasing = TRUE),]

# Predict on Train data
# Predict on Train data 
pred_Train = predict(rf, 
                     train[,setdiff(names(train), "target")],
                     type="response", 
                     norm.votes=TRUE)


# Predicton Test Data

pred_Validation = predict(rf, validation[,setdiff(names(validation),
                                              "target")],
                    type="response", 
                    norm.votes=TRUE)


table(pred_Validation)

library(caret)

result_Train<-confusionMatrix(pred_Train, train$target)

result_Train

result_Train$byClass[7]

result_Validation<-confusionMatrix(pred_Validation, validation$target)

result_Validation

result_Validation$byClass[7]


#Using the model built, predict the values for test data
pred_Test  =  predict(rf, test_data_mod,
                      type="response", norm.votes=TRUE)  

#Write the output file

output<-data.frame(test_data$ID,pred_Test)

colnames(output)<-c("ID","prediction")

summary(output)

write.csv(output,file="Samplesubmission3.csv")


