#Clear the environment 

rm(list=ls(all=TRUE))

setwd("D:/INSOFE/CSE7305c/CUTe 3")

#Read the input data that is given

company_data<-read.csv("train.csv",header = T)

#Read the test data that is given

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

#Impute the data 

library(DMwR)
library(caret)

company_data[is.na(company_data)]<-0

company_data_final<-company_data

test_data[is.na(test_data)]<-0

test_data_std<-test_data

company_data_final$target<-as.factor(as.character(company_data_final$target))

company_data_final$targetlabel <- ifelse(company_data_final$target == "0", "g", "h")

# Divide the data into test and validation
set.seed(123)

train_RowIDs = createDataPartition(company_data_final$target,p=0.8,list=F)
train = company_data_final[train_RowIDs,]
validation= company_data_final[-train_RowIDs,]
test<-test_data_std
rm(train_RowIDs)

#Build an ensemble model with xgboost
install.packages("xgboost")
library(xgboost)

train_matrix <- xgb.DMatrix(data = as.matrix(train[, !(names(train) %in% c("target", "targetlabel"))]), 
                            label = as.matrix(train[, names(train) %in% "target"]))

validation_matrix <- xgb.DMatrix(data = as.matrix(validation[, !(names(validation) %in% c("target", "targetlabel"))]), 
                           label = as.matrix(validation[, names(validation) %in% "target"]))

xgb_model_basic <- xgboost(data = train_matrix, max.depth = 2, eta = 1, nthread = 2, nround = 500, objective = "binary:logistic", verbose = 1, early_stopping_rounds = 10)

xgb.save(xgb_model_basic, "xgb_model_basic")

rm(xgb_model_basic)

xgb_model_basic <- xgb.load("xgb_model_basic")


basic_preds <- predict(xgb_model_basic, validation_matrix)

#Choosing the cut off
basic_preds_labels <- ifelse(basic_preds < 0.5, 0, 1)

library(caret)
result<-confusionMatrix(basic_preds_labels, validation$target)

F1<-result$byClass[7]

params_list <- list("objective" = "binary:logitraw",
                    "eta" = 0.1,
                    "early_stopping_rounds" = 10,
                    "max_depth" = 6,
                    "gamma" = 0.5,
                    "colsample_bytree" = 0.6,
                    "subsample" = 0.65,
                    "eval_metric" = "logloss",
                    "silent" = 1)

xgb_model_with_params <- xgboost(data = train_matrix, params = params_list, nrounds = 500, early_stopping_rounds = 20)

basic_params_preds <- predict(xgb_model_with_params, validation_matrix)

basic_params_preds_labels <- ifelse(basic_params_preds < 0.5, 0, 1)

result_bf_tuning<-confusionMatrix(basic_params_preds_labels, validation$target)

F1_bf_tuning<-result_bf_tuning$byClass[7]

#Variable Importance

variable_importance_matrix <- xgb.importance(feature_names = colnames(train_matrix), model = xgb_model_with_params)

xgb.plot.importance(variable_importance_matrix)

sampling_strategy <- trainControl(method = "repeatedcv", number = 5, repeats = 2, verboseIter = F, allowParallel = T)

param_grid <- expand.grid(.nrounds = 40, .max_depth = c(2, 4, 6), .eta = c(0.1, 0.3),
                          .gamma = c(0.6, 0.5, 0.3), .colsample_bytree = c(0.6, 0.4),
                          .min_child_weight = 1, .subsample = c(0.5, 0.6, 0.9))

xgb_tuned_model <- train(x = train[ , !(names(train) %in% c("targetlabel", "target"))], 
                         y = train[ , names(train) %in% c("targetlabel")], 
                         method = "xgbTree",
                         trControl = sampling_strategy,
                         tuneGrid = param_grid)

xgb_tuned_model$bestTune

plot(xgb_tuned_model)

tuned_params_preds <- predict(xgb_tuned_model, validation[ , !(names(validation) %in% c("targetlabel", "target"))])

result_Validation<-confusionMatrix(tuned_params_preds, validation$targetlabel)

F1_Validation<-result_Validation$byClass[7]

tuned_params_preds_Test <- predict(xgb_tuned_model, test[ , !(names(test) %in% c("targetlabel", "target"))])

#Using the model built, predict the values for test data

basic_params_preds_labels <- ifelse(tuned_params_preds_Test == "g", 0, 1)

#Write the output file

output<-data.frame(test_data$ID,basic_params_preds_labels)

colnames(output)<-c("ID","prediction")

table(output$prediction)

write.csv(output,file="Samplesubmission2.csv")

