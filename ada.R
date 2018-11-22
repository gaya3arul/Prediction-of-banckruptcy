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

library(DMwR)
library(caret)

company_data[is.na(company_data)]<-0

company_data_final<-company_data

test_data[is.na(test_data)]<-0

test_data_std<-test_data

company_data_final$target<-as.factor(as.character(company_data_final$target))

# Divide the data into train and validation
set.seed(123)

train_RowIDs = createDataPartition(company_data_final$target,p=0.8,list=F)
train = company_data_final[train_RowIDs,]
validation= company_data_final[-train_RowIDs,]
rm(train_RowIDs)

#Exclude target variable from train/validation data before building the model
train_Data_wo_target <- train[,-which(names(train) %in% c("target"))]
validation_Data_wo_target <- validation[,-which(names(train) %in% c("target"))]

#Build ensemble model using adaboost 
install.packages("ada")
library(ada)

model_ada = ada(x = train_Data_wo_target, 
            y = train$target, 
            iter=400, loss="exponential", type= "discrete", nu= 0.45)

pred_Train  =  predict(model_ada, train_Data_wo_target)  

#Using the model built, predict the values for validation data
pred_Validation = predict(model_ada, validation_Data_wo_target) 

#Check the accuracy on train/validation data
cm_Train = table(train$target, pred_Train)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
accu_Train

cm_Validation = table(validation$target, pred_Validation)
accu_Validation= sum(diag(cm_Validation))/sum(cm_Validation)
accu_Validation

#Print the confusion Matrix

conf_matrix_Train <- table(train$target, pred_Train)

conf_matrix_Train

recall_Train <- conf_matrix_Train[2, 2]/sum(conf_matrix_Train[2, ])

precision_Train<-conf_matrix_Train[2,2]/sum(conf_matrix_Train[,2])

F1_Train <- (2 * precision_Train * recall_Train) / (precision_Train + recall_Train)
F1_Train

conf_matrix_Validation <- table(validation$target, pred_Validation)

conf_matrix_Validation

recall_Validation <- conf_matrix_Validation[2, 2]/sum(conf_matrix_Validation[2, ])

precision_Validation<-conf_matrix_Validation[2,2]/sum(conf_matrix_Validation[,2])

F1_Validation <- (2 * precision_Validation * recall_Validation) / (precision_Validation + recall_Validation)
F1_Validation

#Using the model built, predict the values for test data
pred_Test  =  predict(model_ada, test_data_std)  

plot(model_ada)

#Write the output file

output<-data.frame(test_data$ID,pred_Test)

summary(output)

colnames(output)<-c("ID","prediction")

write.csv(output,file="Samplesubmission.csv")

