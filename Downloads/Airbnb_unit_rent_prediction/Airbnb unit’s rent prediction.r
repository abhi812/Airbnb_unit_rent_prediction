#IDA-DSA Final project # Group-35

#Installing and recalling packages

install.packages("e1071")
install.packages("pls")
install.packages("lattice")
install.packages("mice")
install.packages("ggplot2")
install.packages("s20x")
install.packages("glmnet")
install.packages("corrplot")
install.packages("MASS")
install.packages("VIM")
install.packages("VIF")
install.packages("car")
install.packages(h20)
library(h2o)
library(car)
library(VIF)
library(MASS)
library(caTools)
library(lattice)
library(mice)
library(ggplot2)
library(caret)
library(s20x)
library(glmnet)
library(pls)
library(e1071)
library(corrplot)
library(MASS)
library(Metrics)
library(VIM)

#Reading Data file.
data<-read.csv("A:/Master_of_Science_Data/3rd_Semester_Fall_2018/Intelligent Data Analytics (DSA_5103)/Project/Data set/train.csv")

#Inspecting for number of missing values in each variable.
for (i in c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29))
{
  data[, i][which(data[, i] == "")] <- NA
  print(paste(i, sum(is.na(data[, i]))))
}

#Selecting variables which has fewer missing values.
newData<- data[,c(2, 3, 4, 6, 7, 8, 9, 10, 11, 14, 15, 16, 18, 20, 21, 23, 24, 25, 28, 29)]
#Removing '%' simbol from variable "host_response_rate".
newData$host_response_rate <- as.numeric((sub("%","",newData$host_response_rate, fixed = TRUE)))

#Predictive Mean Matching Imputation.
imputed <- mice(newData, m = 1, method = 'rf', seed = 27)
summary(imputed)
NewData_imputed <- complete(imputed, 1)
View(NewData_imputed)

#Calculating number of days since perticular unit is hosting guests.
Data_again <- data$host_since
Data_again$host_since <- '2018-12-01'
Host_Since_Days <- as.Date(Data_again$host_since, format = "%Y-%m-%d") - as.Date(data$host_since, "%Y-%m-%d")
data <- cbind(data, Host_Since_Days)

NewData_imputed <- cbind(NewData_imputed, data$Host_Since_Days)

#Removing observations which has missing values in any of variable.
NewData_imputed <- na.omit(NewData_imputed)

#Checking structure of dataset.
str(NewData_imputed)

#Splitting data into train and test samples.
splittingratio <- sample.split(NewData_imputed, SplitRatio = 0.7)
trainData<- subset(NewData_imputed, splittingratio==TRUE)
testData<- subset(NewData_imputed, splittingratio==FALSE)
#Removing Predicting variable from test dataset.
testData<- subset(testData, select = -log_price)

trainData_numeric <- subset(trainData, select = -c(property_type, room_type, bed_type, cancellation_policy, cleaning_fee, city, host_has_profile_pic, host_identity_verified, instant_bookable, neighbourhood))
trainData_qualitative <- subset(trainData, select = c(property_type, room_type, bed_type, cancellation_policy, cleaning_fee, city, host_has_profile_pic, host_identity_verified, instant_bookable, neighbourhood))

#Models
reglm <- lm(data = trainData, log_price~ .)
summary(reglm)
#Adjusted R-squared:  0.6502 
testinglm <- predict(reglm, trainData)
#Calculating RMSE value
rmse(trainData$log_price, testinglm)
#  0.4213543
AIC(reglm)
# 56001.23
BIC(reglm)
# 61865.57

#no improvement, dont do that, just for academic purpose

fit1<- stepAIC(fit1, direction = "both")
fit2<- reglm
summary(fit2)
ncvTest(fit2)

#histogram of residuals
qplot(fit2$resid) + geom_histogram(binwidth=0.15)

#qq plot of residuals
qqnorm(fit2$resid)
qqline(fit2$resid)

residualPlots(fit2)
confint(fit2,conf.level=.95)
ciReg(fit2)
plot(fit2)
anova(fit2)

#index leverage plot
plot(hatvalues(fit2),col = "black", pch = 21, bg = "red")      #index plot of leverages
abline(h=2*20/50000)
hatvalues(fit2)[hatvalues(fit2)>0.00084]
#plot residuals vs. hatvalues
plot(hatvalues(fit2),fit2$residuals,col = "black", pch = 21, bg = "red")    #leverages and residuals
abline(h=0,v=2*20/50000)
#standardized residuals (index plot) ----- 
plot(rstandard(fit2),col = "black", pch = 21, bg = "red")      # index standardized residual plot
abline(h=c(-2,2), lty = 2)
#studentized residals vs. Cook's D
plot(cooks.distance(fit2),rstudent(fit2),col = "black", pch = 21, bg = "red")
# several influence measures
#influence.measures(fit2)
# influence plot from car package
influencePlot(fit2)

#############Performing Ridge regression.#################################################################
#Calculating lambda sequence
l <- 10^seq(0.00001, 1, length = 10)
#set.seed(2707)
regridge <- train(data = trainData, log_price ~ ., method = "glmnet", trControl = trainControl("cv", number = 10), tuneGrid = expand.grid(alpha = 0, lambda = l))
#Predicting the log_price 
summary(regridge)
testingridge <- predict(regridge, testData)
#Calculating RMSE value
rmse(trainData$log_price, testingridge)
# 0.7972118

############Performing Lasso regression.##################################################################
reglasso <- train(data = trainData, log_price ~ ., method = "glmnet", trControl = trainControl("cv", number = 10), tuneGrid = expand.grid(alpha = 1, lambda = l))
summary(reglasso)
#Predicting the log_price 
testinglasso <- predict(reglasso, testData)
#Calculating RMSE value
rmse(trainData$log_price, testinglasso)
#0.7172321

##########Performing Elastic net regression############################################################.
regelastic <- train(data = trainData, log_price ~ ., method = "glmnet", trControl = trainControl("cv", number = 10), tuneGrid = expand.grid(alpha = seq(0, 1, 10), lambda = l))
summary(regelastic)
#Predicting the log_price 
testingelastic <- predict(regelastic, testData)
#Calculating RMSE value
rmse(trainData$log_price, testingelastic)
# 0.7972118

###########Performing PLS.############################################################################

#using randomly all principal components to model data
regpls <- plsr(log_price ~ ., data = trainData, method = "oscorespls", validation = "CV")

summary(regpls)
#Predicting the ViolentCrimesPerPop using train data.
testingpls <- predict(regpls, trainData)
#Calculating RMSE value
rmse(trainData$log_price, testingpls)
#2.501807e+15

plot(RMSEP(regpls),legendpos="topright")
#Form the plot, we see 4 components are sufficient to explain most of the variance
regpls1 <- plsr(data =trainData, log_price ~ .,4, method = "oscorespls", validation = "CV")
summary(regpls1)
plot(regpls, ncomp = 4, asp = 1, line = TRUE)
plot(regpls, plottype = "scores", comps = 1:3)

testingpls1 <- predict(regpls1, trainData)
#Calculating RMSE value
rmse(trainData$log_price, testingpls1)

#0.6519792

############Random forest with h2o ####################################################################

#creating an H2o cloud using all the available threads and 2G size
h2o.init(nthreads = -1, max_mem_size = "2G")  

# Clean slate - just in case the cluster was already running
h2o.removeAll()

df<- h2o.importFile(path = normalizePath("newdata.csv"))
head(df)

# splitting the data in to trianing, validation, and test sets
# we create splits of 60% and 20%;H2O will create one more split of 1-(sum of these parameters)
# so we will get 0.6 / 0.2 / 1 - (0.6+0.2) = 0.6/0.2/0.2
splits<- h2o.splitFrame(df, c(0.6,0.2), seed = 1234)
train <- h2o.assign(splits[[1]], "train.hex")
valid <- h2o.assign(splits[[2]], "valid.hex")
test<- h2o.assign(splits[[3]], "test.hex")

# lets lok at the head of the trianing dataset
train[1:5,]

# running random forest model. Note: this take seconds instead of forever :D
rf1 <- h2o.randomForest(training_frame = train, validation_frame = valid, x=2:21, y=1, model_id = "rf_covType_v1", ntrees = 200, stopping_rounds = 2, score_each_iteration = T, seed = 1000000)
summary(rf1)
rf1@model$validation_metrics
perf <- h2o.performance(rf1, test)
perf
###########################  End  ##################################################################

