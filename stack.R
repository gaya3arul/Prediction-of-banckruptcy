#Clear the environment 

rm(list=ls(all=TRUE))

#Read the input data that is given

setwd("D:/INSOFE/CSE7305c/CUTe 3")

# Load required libraries
library(vegan)
library(infotheo)
library(C50)
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
head(company_data_mod)

# Divide the data into train and validation
set.seed(123)

train_RowIDs = createDataPartition(company_data_mod$target,p=0.9,list=F)
train = company_data_mod[train_RowIDs,]
validation= company_data_mod[-train_RowIDs,]
test=test_data_mod

# Check how records are split with respect to target attribute.
table(train$target)
table(validation$target)
rm(company_data_mod)

#----------------Ensemble:Stacking-------------------- 

# Build CART model on the training dataset
cart_Model = rpart(target ~ ., train, method = "class")
summary(cart_Model)


# Build C5.0 model on the training dataset
c50_Model = C5.0(target ~ ., train, rules = T)
summary(c50_Model)

# Build Logistic regression on the training dataset
glm_Model = glm(target ~ ., train, family = binomial(link='logit'))
summary(glm_Model)



#---------Predict on Train Data----------

# Using CART Model predict on train data
cart_Train = predict(cart_Model, train, type = "vector") 
table(cart_Train)

# if we choose type=vector, then replace 1 with 0 and 2 with 1
cart_Train = ifelse(cart_Train == 1, 0, 1)
table(cart_Train)

# Using C5.0 Model predicting with the train dataset
c50_Train = predict(c50_Model, train, type = "class")
c50_Train = as.vector(c50_Train)
table(c50_Train)

library(ROCR)
# Using GLM Model predicting on train dataset
glm_Train = predict(glm_Model, train, type = "response")

# it gives probabilities, so we #need to convert to 1's and 0's; 
# if >0.5 show as 1 or else show as 0.
glm_Train = ifelse(glm_Train > 0.5, 1, 0) 
table(glm_Train)

# Combining training predictions of CART, C5.0 & Log Regression together
train_Pred_All_Models = data.frame(CART = cart_Train, 
                                   C50 = c50_Train,
                                   GLM = glm_Train)
train_Pred_All_Models = data.frame(sapply(train_Pred_All_Models, 
                                          as.factor))

# or first use "apply" then type data_ensemble = data.frame(data_ensemble)
str(train_Pred_All_Models)
summary(train_Pred_All_Models)


# Viewing the predictions of each model
table(train_Pred_All_Models$CART) #CART 
table(train_Pred_All_Models$C50)  #C5.0
table(train_Pred_All_Models$GLM)  #Logistic Regression
table(train$target) #Original Dataset DV


# Adding the original DV to the dataframe
train_Pred_All_Models = cbind(train_Pred_All_Models, target = train$target)

# Ensemble Model with GLM as Meta Learner
str(train_Pred_All_Models)
head(train_Pred_All_Models)

ensemble_Model = glm(target ~ ., train_Pred_All_Models, 
                     family = binomial)
summary(ensemble_Model)

# Check the "ensemble_Model model" on the train data
ensemble_Train = predict(ensemble_Model, train_Pred_All_Models, 
                         type = "response")
ensemble_Train = ifelse(ensemble_Train > 0.5, 1, 0)
table(ensemble_Train)

cm_Ensemble = table(ensemble_Train, train_Pred_All_Models$target)
sum(diag(cm_Ensemble))/sum(cm_Ensemble)

cm_c50 = table(c50_Train, train_Pred_All_Models$target)
sum(diag(cm_c50))/sum(cm_c50)


#---------Predict on Validation Data----------

# Using CART Model prediction on validation dataset
cart_Validation = predict(cart_Model, validation, type="vector")
cart_Validation = ifelse(cart_Validation == 1, 0, 1)

cm_CART = table(cart_Validation, validation$target)
sum(diag(cm_CART))/sum(cm_CART)

# Using C50 Model prediction on validation dataset 
c50_Validation = predict(c50_Model, validation, type = "class")
c50_Validation = as.vector(c50_Validation)

cm_C50 = table(c50_Validation, validation$target)
sum(diag(cm_C50))/sum(cm_C50)

# Using GLM Model prediction on validation dataset
glm_Validation = predict(glm_Model, validation, type="response")
glm_Validation = ifelse(glm_Validation > 0.5, 1, 0)

cm_Glm = table(glm_Validation, validation$target)
sum(diag(cm_Glm))/sum(cm_Glm)


###########################################################

# Combining validation predictions of CART, C5.0 & Log Regression together 
validation_Pred_All_Models = data.frame(CART = cart_Validation, 
                                  C50 = c50_Validation, 
                                  GLM = glm_Validation) 
rm(cart_Validation, c50_Validation, glm_Validation)

validation_Pred_All_Models = data.frame(sapply(validation_Pred_All_Models, as.factor))
str(validation_Pred_All_Models)
head(validation_Pred_All_Models)

# Check the "glm_ensemble model" on the validation data
ensemble_Validation = predict(ensemble_Model, validation_Pred_All_Models, type = "response")
ensemble_Validation = ifelse(ensemble_Validation > 0.5, 1, 0)
table(ensemble_Validation)

cm_Ensemble = table(ensemble_Validation, validation$target)
sum(diag(cm_Ensemble))/sum(cm_Ensemble)

#---------Predict on Test Data----------

# Using CART Model prediction on test dataset
cart_Test = predict(cart_Model, test, type="vector")
cart_Test = ifelse(cart_Test == 1, 0, 1)

# Using C50 Model prediction on test dataset 
c50_Test = predict(c50_Model, test, type = "class")
c50_Test = as.vector(c50_Test)

# Using GLM Model prediction on test dataset
glm_Test = predict(glm_Model, test, type="response")
glm_Test = ifelse(glm_Test > 0.5, 1,0 )

###########################################################

# Combining test predictions of CART, C5.0 & Log Regression together 
test_Pred_All_Models = data.frame(CART = cart_Test, 
                                        C50 = c50_Test, 
                                        GLM = glm_Test) 
rm(cart_Test, c50_Test, glm_Test)

test_Pred_All_Models = data.frame(sapply(test_Pred_All_Models, as.factor))
str(test_Pred_All_Models)
head(test_Pred_All_Models)

# Check the "glm_ensemble model" on the validation data
ensemble_Test = predict(ensemble_Model, test_Pred_All_Models, type = "response")
ensemble_Test = ifelse(ensemble_Test > 0.5, 1, 0)
table(ensemble_Test)

#Write the output file

output<-data.frame(test_data$ID,ensemble_Test)

colnames(output)<-c("ID","prediction")

table(output$prediction)

write.csv(output,file="Samplesubmission4.csv")




