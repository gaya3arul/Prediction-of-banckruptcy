#Clear the environment 

rm(list=ls(all=TRUE))

#Read the input data that is given

setwd("D:/INSOFE/CSE7305c/CUTe 3")

# Load required libraries
library(caret)
library(rpart)

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

company_data_mod[is.na(company_data_mod)]<-0

test_data_mod[is.na(test_data_mod)]<-0

#After imputing, add target colum back to the train data
company_data_mod<-cbind(company_data_mod,target=company_data$target)

company_data_mod$target<-as.factor(as.character(company_data_mod$target))

# Divide the data into train and validation
set.seed(123)

train_RowIDs = createDataPartition(company_data_mod$target,p=0.8,list=F)
train = company_data_mod[train_RowIDs,]
validation= company_data_mod[-train_RowIDs,]
test=test_data_mod

#----------------C50-------------------- 

library(C50)

# Build C5.0 model on the training dataset
c50_Model = C5.0(target ~ ., train, rules = T)
summary(c50_Model)

# Using C5.0 Model predicting with the train dataset
c50_Train = predict(c50_Model, train, type = "class")
c50_Train = as.vector(c50_Train)
table(c50_Train)

# Using C50 Model prediction on validation dataset 
c50_Validation = predict(c50_Model, validation, type = "class")
c50_Validation = as.vector(c50_Validation)

cm_C50 = table(c50_Validation, validation$target)
sum(diag(cm_C50))/sum(cm_C50)


#---------Predict on Test Data----------


# Using C50 Model prediction on test dataset 
c50_Test = predict(c50_Model, test, type = "class")
c50_Test = as.vector(c50_Test)


#Write the output file

output<-data.frame(test_data$ID,c50_Test)

colnames(output)<-c("ID","prediction")

table(output$prediction)

write.csv(output,file="Samplesubmission5.csv")
