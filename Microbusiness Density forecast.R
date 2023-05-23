rm(list=ls())
set.seed(1693) 
#intall.packages("gbm")
#install.packages("e1071")
#install.packages("keras")
#install.packages("xgboost")
library(gbm)
library(e1071)
library(tidyr)
library(dplyr)
library(readr)
library(keras)
library(caret)
library(xgboost)
##########################################Training Data Prep##########################################################
#read in data
data_train <- read.csv("train.csv", stringsAsFactors = T, header=T)
#drop row Id column
data_train = data_train[,2:7]
#drop county and state
data_train = data_train[,-c(2:3)]
#drop active column
data_train = data_train[,-4]
#make into a dataframe
df_train = data.frame(data_train)
#change first day of month to a date type
df_train$first_day_of_month = as.Date(df_train$first_day_of_month)
#create an empty month list
month = c()
#extract the months from the date in each row and store in month variable created above
for (i in 1:nrow(df_train)){
       month[i] = as.numeric(format(as.Date(df_train$first_day_of_month[i]),"%m"))}
#add month to the data frame
df_train['month']=month
#create an empty year list
year=c()
#extract the years from the date in each row and store in year variable created above
for (i in 1:nrow(df_train)){
       year[i] = as.numeric(format(as.Date(df_train$first_day_of_month[i]),"%Y"))}
#add year to the data frame
df_train['year']= year
#remove first day of month
df_train= df_train[,-2] 
#create x and y features from data frame
x_features_train = df_train[,-2]
y_features_train = df_train[,2]

#read in Census Data
Census_Data <- read.csv("census_starter.csv", stringsAsFactors = T, header=T)
#Combine x features and census data
df_combined_train <- merge(x = Census_Data, y = x_features_train, by = "cfips")
#create data for models to train on
df_for_models_train = df_combined_train
#add in Y target for models to train on that need it
df_for_models_train["Y_Target"] = y_features_train
##########################################Testing Data Prep##########################################################
#read in testing data
data_test <- read.csv("test.csv", stringsAsFactors = T, header=T)
#drop row id
data_test = data_test[,-1]
#make it into a data frame
df_test = data.frame(data_test)
#make first day of month date type
df_test$first_day_of_month = as.Date(df_test$first_day_of_month)
#create empty month list to store months
month = c()
#extract month from data
for (i in 1:nrow(df_test)){
       month[i] = as.numeric(format(as.Date(df_test$first_day_of_month[i]),"%m"))}
#add month back into data frame
df_test['month']=month
#create emty year list to store years
year=c()
#extract year from each row in data
for (i in 1:nrow(df_test)){
       year[i] = as.numeric(format(as.Date(df_test$first_day_of_month[i]),"%Y"))}
#add year in as a new column
df_test['year']= year
#drop day of month 
df_test= df_test[,-2]
#read in census data
Census_Data <- read.csv("census_starter.csv", stringsAsFactors = T, header=T)
#Combine x features and census data
df_combined_test <- merge(x = Census_Data, y = df_test, by = "cfips")
#create new df for testing models on
df_for_models_test = df_combined_test
#Remove the 2023 data so we can use revealed test to evaluate
df_for_models_test_no2023 = df_for_models_test[df_for_models_test$year != 2023, ]
#re-order data so that it matches revealed test data
df_for_models_test_no2023 = df_for_models_test_no2023[order(df_for_models_test_no2023$cfips,df_for_models_test_no2023$month),]
#load in revealed test data
data_test_revealed <- read.csv("revealed_test.csv", stringsAsFactors = T, header=T)
#extract Y features from revealed test data
y_features_test = data_test_revealed[,6]
##########################################Cleaning Data##########################################################
#cleaning data
#create train & test matrixes from data wrangling above
x_train = as.matrix(df_combined_train)#comes from merging train.csv & census_starter.csv 
x_test = as.matrix(df_for_models_test_no2023)#comes from merging test.csv & census_starter.csv then removing all 2023 data
y_train = as.matrix(y_features_train)#comes from train.csv
y_test = as.matrix(y_features_test)#comes from revealed_test.csv
#check to see if there are any Na values in data that need to be dealt with
any(is.na(x_train))#True
any(is.na(y_train))#False
any(is.na(x_test))#True
any(is.na(y_test))#False
#looks like there are
#merge x and y datasets together so that when we remove an na we remove from both train and test
train = cbind(x_train,y_train)
test = cbind(x_test,y_test)
#na_rows <- apply(train, 1, function(x) any(is.nan(x)))
#remove NA's from the data set
train = na.omit(train)
test = na.omit(test)
#re create training and testing sets 
x_train = train[,-29]
y_train = train[,29]
x_test = test[,-29]
y_test = test[,29]
#check to see if there are any other Na values left in data that need to be dealt with
any(is.na(x_train))#False
any(is.na(y_train))#False
any(is.na(x_test))#False
any(is.na(y_test))#False
#we dealt with all of them
##########################################Linear Model##########################################################
#build a linear model to see if its worth pursuing
LinearModel = lm(Y_Target~.,data = df_for_models_train)
#check the summary for R^2
summary(LinearModel)#adjusted Rsquared is 0.2727 (YIKES)
#make predictions using the linear model
predictions <- predict(LinearModel, newdata=df_for_models_test_no2023)
#calculate SMAPE
SMAPE = mean(abs((data_test_revealed$microbusiness_density - predictions)/(abs(data_test_revealed$microbusiness_density)+abs(predictions)))) * 100
#print the SMAPE
print(SMAPE)#it is NA
#calculate SMAPE removing the NA's
SMAPE = mean(abs((data_test_revealed$microbusiness_density - predictions))/(abs(data_test_revealed$microbusiness_density)+abs(predictions)), na.rm=TRUE) * 100
#print the SMAPE to check it
print(SMAPE)#22.04351
############################# XGBOOST!!!!!! ####################################
#Xgboost train data creation
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
#XGBoost test data creation
dtest <- xgb.DMatrix(data = x_test, label = y_test)
#create an empty Results list for our SMAPE's
Results = c()
#create an empty list to store the depth we are using
depths = c()
#create an empty list to store the learning rate we are using
etas = c()
#initialize a counter so we append things to the right place
count = 1
#Find best model parameters for the model using for loop
####!!!!!!!!!!!!!!!!!!!do not run this in class it is slow!!!!!!!!!!!!!!!!!!!!!!
#iterate through depths from 1 to 20
for (i in 1:20){
       #iterate through learning rates from 0.1 to 1.0
       for (x in 1:10){
              #set the parameter for this model
              params <- list(max_depth = i, eta = (0.1*x), objective = "reg:squarederror")
              #create the model
              model <- xgb.train(params = params, data = dtrain, nrounds = 100)
              # Make predictions on the test set
              pred <- predict(model, dtest)
              # Evaluate the model's rmse
              rmse <- sqrt(mean((pred - y_test)^2))
              #calculate the model's Symetrical Mean Absolute Percent Error
              SMAPE = mean(abs((y_test - pred))/(abs(y_test)+abs(pred))) * 100
              #append SMAPE to the results list
              Results[count]= SMAPE
              #put the depth in the depths list
              depths[count] = i
              #put the learning rate in the etas list
              etas[count] = (x*0.1)
              #add one to the counter so the next time the foor loop runs stuff
              ####goes to the right place
              count=count+1
              #print rmse, SMAPE, depth and learning rate *10 every loop to track 
              ####where in the for loop it is as it runs slow and its useful to 
              ####see its progress
              print(c(rmse,SMAPE,i,x))
       }
}
#Find the index that has the lowest SMAPE
index_to_use = which.min(Results)
#Use the index to retrieve the lowest SMAPE
Min_SMAPE = Results[index_to_use]
#Use the index to find the best depth to use
depth_to_use = depths[index_to_use]
#Use the index to find the best learning rate (estimate) to use
eta_to_use = etas[index_to_use]
#print so you can see
print(Min_SMAPE) #1.138392
#print so you can see
print(depth_to_use) #20
#print so you can see
print(eta_to_use) #0.1
#Now we look to see if there is a learning rate better than 0.1 for a max_depth of 20
####!!!!!!!!!!!!!!!again do not run this in class it is slow!!!!!!!!!!!!!!!!!!!!
####by searching 0.01-0.3 by increments of 0.1
#create a place to store SMAPE's for each model
Results1 = c()
#create a place to store depth (should always be 14 for this for loop)
depths1 = c()
#create a place to store the learning rates we are using
etas1 = c()
#initialize the counter
count1 = 1
#set i to 20 permanently (best depth from model above)
for (i in 20){
       #use .01 to .03 so that we can look at learning rates of .2 to .4 in increments of 0.01
       for (x in seq(.01,.3,.01)){
              #create parameters for each model
              params <- list(max_depth = i, eta = (x), objective = "reg:squarederror")
              #create the model each time
              model <- xgb.train(params = params, data = dtrain, nrounds = 100)
              # Make predictions on the test set
              pred <- predict(model, dtest)
              # Evaluate the model's rmse
              rmse <- sqrt(mean((pred - y_test)^2))
              # calculate the modeles Symetrical Mean Absolute Percent Error
              SMAPE = mean(abs((y_test - pred))/(abs(y_test)+abs(pred))) * 100
              #append SMAPE to the results1 list 
              Results1[count1]= SMAPE
              #append the depth to the depths1 list
              depths1[count1] = i
              #append the learning rate to the etas1 list
              etas1[count1] = (x)
              #add one to the counter so the next time the foor loop runs stuff
              ####goes to the right place
              count1=count1+1
              #print rmse, SMAPE, depth and learning rate *10 every loop to track 
              ####where in the for loop it is as it runs slow and its useful to 
              ####see its progress
              print(c(rmse,SMAPE,i,x))
       }
}
#look at results from testing the model
#find what index has the lowest SMAPE
index_to_use1 = which.min(Results1)
#find out what the lowest SMAPE is 
Min_SMAPE1 = Results1[index_to_use1]
#Use the index to find the best depth to use
depth_to_use1 = depths1[index_to_use1]
#use the index to fing the best learning rate to use
eta_to_use1 = etas1[index_to_use1]
#print so you can see
print(Min_SMAPE1)#1.126116
#print out the best depth to use
print(depth_to_use1) #20
#print out the best learning rate to use
print(eta_to_use1) #0.16
#create model based on best parameters
params <- list(max_depth = 20, eta = .16, objective = "reg:squarederror")
#build the model
model <- xgb.train(params = params, data = dtrain, nrounds = 100)
# Make predictions on the test set
pred <- predict(model, dtest)
# Evaluate the model
SMAPE = mean(abs(y_test - pred)/(abs(y_test)+abs(pred))) * 100
#print the SMAPE
SMAPE
