
# title: "ML_Project_Kaggle"
# author: "Mehul Patel"

#training data and required packages
rm(list = ls())

#Loading required packages which will be used to get the data set in a datafrmae.
packages <- c("jsonlite", "dplyr", "purrr")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)

#Loading packages for further analysis of the data.
libs <- c("ggmap", "ggplot2", "lubridate", "nnet", "data.table", "dplyr", "tree", "randomForest", "glmnet", "caret")
lapply(libs, require,  character.only = T)

#Importing training data set and converting it into a datafrmae.
data_train <- fromJSON("filepath//train.json")
vars <- setdiff(names(data_train), c("photos", "features"))
data_train <- map_at(data_train, vars, unlist) %>%
        tibble::as_tibble(.)
head(data_train,1)

#Extracting required meta-data from the data and attaching it to actual data}
##Counting the #features for each entry and  attaching it to a data frame:
data_train <- mutate(data_train, feature_count = unlist(lapply(data_train$features, length)))


#I think it's necessary to check the influennce of the date and time on which each listing was added to the website.
#Extracting month and hours  from "created" column:
date_time <- strptime(data_train$created, format = "%Y-%m-%d %H:%M:%S")
data_month <- month(date_time)
data_hour <- hour(date_time)
data_train <- cbind(data_train, data_month, data_hour)

#Plotting some features of the data set to understand the story behind data:

##Exploratory Data Analysis:
#Based on #bathrooms:

ggplot(data = data_train, aes(x = factor(bathrooms), color = interest_level, fill = interest_level)) + 
  geom_bar() +
  ggtitle("Plot of No.of bathrooms considering interest level")

#Based on #bedrooms:  
ggplot(data = data_train, aes(x = factor(bedrooms), color = interest_level, fill = interest_level)) + 
  geom_bar() +
  ggtitle("Plot of No.of bedrooms considering interest level")

ggplot(data = data_train, aes(x = bedrooms, color = interest_level, fill = interest_level)) + 
  geom_histogram(binwidth = 1)


#Checking plots on the basis of date-time on which the listings were created:
ggplot(data_train, aes(x = factor(data_month), fill = interest_level)) +
  geom_bar()

ggplot(data_train, aes(x = factor(data_hour), fill = interest_level)) +
  geom_bar()


#Distribution of interest levels:
table(data_train$interest_level)

#Modeling Process 1: Multinomial Logistic Regression

#################################################################
####-------Fitting Multinomial Logistic Regression Model:----####
#################################################################

#Converting InterestLevel feature into factor:
unique(data_train$interest_level)
interest_factor_2 <- rep(NA, nrow(data_train))

data_train <- cbind(data_train, interest_factor_2)
data_train$interest_factor_2[data_train$interest_level == "low"] <- 0
data_train$interest_factor_2[data_train$interest_level == "medium"] <- 1
data_train$interest_factor_2[data_train$interest_level == "high"] <- 2
data_train$interest_factor_2 <- factor(data_train$interest_factor_2)

#As I am performing Multinomial LR, it's preferrable to convert one of three classes of the outcome into reference. LR works better when outcome has only two classes. So, in this model, I am making one class of outcome as a reference. That is, in o/p we'll have only two classes but while calculating log-odds of probability, we'll have to consider the reference class.

#Modeling Process 2: Training the Multinomial LR model:

data_train$interest_factor_mult <- relevel(data_train$interest_factor_2, ref = "0")

#Fitting the Multinomial LR:
start_MLR <- Sys.time()

LR_fit_1 <- multinom(interest_factor_mult ~ bathrooms + bedrooms + price + feature_count + data_month + data_hour, data = data_train)

end_MLR <- Sys.time()
RT_MLR <- end_MLR - start_MLR

#Low interest level is the base (or reference level == 0).
summary(LR_fit_1)
# 
# #Cross-Validation for multinomial LR:
# data_train_mat <- subset(data_train, select = c(bathrooms, bedrooms, price, feature_count, 
#                                                  data_month, data_hour, interest_factor_2)) %>%
#                    as.matrix()
# class(data_train)
# 
# LR_fit_cv <- glmnet(x = data_train_mat[c(1:1000), c(1:6)], y = data_train_mat[c(1:1000), 7], family = "multinomial", type.multinomial = "grouped")
# 
# plot(LR_fit_cv)
# plot(LR_fit_cv, xvar = "lambda", label = T, type.coef = "2norm")

# 
# x <-  data_train_mat[, c(1:6)] 
# y <-  data_train_mat[, 7]
# 
# LR_fit_cv_1 <- cv.glmnet(as(x, "dgCMatrix"), y, family = "multinomial", type.multinomial = "grouped", parallel = T, nfolds = 10)
# 
# plot(LR_fit_cv_1)
# 
# cv_pred <- predict(LR_fit_cv_1, newx = as(data_train_mat[c(1001:2000), c(1:6)], "dgCMatrix"), s = 0.04033, type = "class")

#Modeling Process 3: Making predictions on training data using Multinomial LR model:

#Making predictions on training data and checking accuracy of the model:
pred_train <- predict(LR_fit_1, data_train)
pred_train[1:20]

#Creating a confusion matrix to where model predicts right/wrong:
cm <- table(predict(LR_fit_1), data_train$interest_factor_mult)
cm

#Following command calculates accuracy of the model.
acc_1 <- mean(predict(LR_fit_1) == data_train$interest_factor_2)
acc_1

#Making predictions with probabilities:
prob_pred <- predict(LR_fit_1, data_train, type = "prob") %>% 
             as.data.frame() %>% 
             setNames(.,c('Low','Medium','High'))          
  
head(prob_pred)

#Modeling Process 4: Importing testing data and manipulating it to bring it in same format as training data:

data_test <- fromJSON("Filepath\\test.json")
vars_2 <- setdiff(names(data_test), c("photos", "features"))
data_test <- map_at(data_test, vars_2, unlist) %>%
  tibble::as_tibble(.)
head(data_test,1)

##Counting the #features for each entry and  attaching it to a data frame:
feature_count <- unlist(lapply(data_test$features, length))
length(feature_count)
data_test <- cbind(data_test, feature_count)

#Converting "created" column into Date-Time:
date_time <- strptime(data_test$created, format = "%Y-%m-%d %H:%M:%S")
data_month <- month(date_time)
data_hour <- hour(date_time)
data_test <- cbind(data_test, data_month, data_hour)

#Modeling Process 5: Making predictions on testing data using Multinomial LR model:

#Making predictions on testing data:
pred_train <- predict(LR_fit_1, data_test)
pred_train[1:20]

#Making predictions with probabilities:
prob_pred <- predict(LR_fit_1, data_test, type = "prob")
colnames(prob_pred) <- c("Low", "Medium", "High")
rownames(prob_pred) <- data_test$listing_id
head(prob_pred)

#Modeling Process 6: Sentiment Analysis on Multinomial LR model:
library("syuzhet")

sentiment <- get_nrc_sentiment(data_train$description)
sort(colSums(sentiment))

data_train_2 <- cbind(data_train, sentiment)

#Fitting the Multinomial LR:
start_MLR_SA <- Sys.time()

LR_fit_2 <- multinom(interest_factor_mult ~ bathrooms + bedrooms + price + feature_count + data_month + data_hour + positive + negative + trust, data = data_train_2)

end_MLR_SA <- Sys.time()
RT_MLR_SA <- end_MLR_SA - start_MLR_SA

#Making predictions on training data set:
pred_train_2 <- predict(LR_fit_2, data_train_2)

#Creating a confusion matrix to where model predicts right/wrong:
cm_2 <- table(predict(LR_fit_2), data_train_2$interest_factor_mult)
cm_2

#Following command calculates accuracy of the model.
acc_2 <- mean(predict(LR_fit_2) == data_train_2$interest_factor_2)
acc_2

#Increase in %accuracy (which is negligible):
paste0("Change in accuracy = ", round((acc_2 - acc_1)*100), "%")

#Modeling Process 7: Random Forest Algorithm:

#Subsetting the data for Random Forest Algorithm:
data_train_4 <- subset(data_train, select = c(bathrooms, bedrooms, price, feature_count, data_month, data_hour, 
                                              interest_factor_2))


#Applying RF algorithm:
start_RF <- Sys.time()

RF_1 <- randomForest(x = data_train_4[,c(1:6)], y = data_train_4$interest_factor_2, ntree = 100, importance = T)

end_RF <- Sys.time()
RT_RF <- end_RF - start_RF

#Predicting classes (i.e. 0/1/2) on training data:
pred_class_RF_1 <- predict(RF_1, newdata = data_train_4, type = "class")

#Creating a confusion matrix to where model predicts right/wrong:
cm_RF_1 <- table(pred_class_RF_1, data_train_4$interest_factor_2)
cm_RF_1

#Following command calculates accuracy of the model.
acc_RF_1 <- mean(pred_class_RF_1 == data_train_4$interest_factor_2)
acc_RF_1

#Predicting probabilities for low, medium, and high interest level:
pred_prob_RF <- predict(RF_1, newdata = data_train_4, type = "prob") %>%
                as.data.frame() %>% 
                setNames(.,c('Low','Medium','High'))

head(pred_prob_RF)

#Modeling Process 8: Sentiment Analysis on Random Forest Model:

#Using sentiments in Random Forest Algorithm and checking the accuracy:
data_train_5 <- subset(data_train_2, select = c(bathrooms, bedrooms, price, feature_count, data_month, data_hour, 
                                              positive, negative, trust, interest_factor_2))

#Applying RF algorithm:
start_RF_SA <- Sys.time()

RF_2 <- randomForest(x = data_train_5[,c(1:9)], y = data_train_5$interest_factor_2, ntree = 100, importance = T)

end_RF_SA <- Sys.time()
RT_RF_SA <- end_RF_SA - start_RF_SA 

#Predicting classes (i.e. 0/1/2) on training data:
pred_class_RF_2 <- predict(RF_2, newdata = data_train_5, type = "class")

#Creating a confusion matrix to where model predicts right/wrong:
cm_RF_2 <- table(pred_class_RF_2, data_train_5$interest_factor_2)
cm_RF_2

#Following command calculates accuracy of the model.
acc_RF_2 <- mean(pred_class_RF_2 == data_train_5$interest_factor_2)
acc_RF_2

#Change in accuracy:
paste0("Change in accuracy = ", round((acc_RF_2 - acc_RF_1)*100), "%")

# getTree(rfobj = RF_1, k = 50, labelVar = T)
# varImpPlot(RF_1)


#Modeling Process 9: Making predictions on testing data using Random Forest model:

#Making predictions with probabilities:
prob_pred_test_RF <- predict(RF_1, data_test, type = "prob")
colnames(prob_pred_test_RF) <- c("Low", "Medium", "High")
rownames(prob_pred_test_RF) <- data_test$listing_id
head(prob_pred_test_RF)

#Scaled Data Experiment: Gives almost the same results as unscaled data:
names(data_train)
data_train_scaled <- as.data.frame(cbind(interest_factor_mult = data_train[,c(19)], scale(data_train[,c(1,2,13,16,17,18)])))
data_train_scaled <- subset(data_train, select = c(bathrooms, bedrooms, price, feature_count, 
                                                   data_month, data_hour)) %>%
                     scale() %>%
                     as.data.frame()

data_train_scaled <- cbind(data_train_scaled, interest_factor_mult = data_train$interest_factor_mult, 
                            interest_factor_2 = data_train$interest_factor_2)

#Fitting the Multinomial LR:
start_MLR_SD <- Sys.time()

LR_fit_2 <- multinom(interest_factor_mult ~ bathrooms + bedrooms + price + feature_count + data_month + data_hour, data = data_train_scaled)

end_MLR_SD <- Sys.time()
RT_MLR_SD <- end_MLR_SD - start_MLR_SD 

pred_train_scaled <- predict(LR_fit_2, data_train_scaled)
pred_train[1:20]

#Creating a confusion matrix to where model predicts right/wrong:
cm_scaled <- table(pred_train_scaled, data_train_scaled$interest_factor_mult)
cm_scaled

#Following command calculates accuracy of the model.
acc_scaled <- mean(pred_train_scaled == data_train_scaled$interest_factor_2)
acc_scaled

#Making predictions with probabilities:
prob_pred_scaled <- predict(LR_fit_2, data_train_scaled, type = "prob") %>% 
             as.data.frame() %>% 
             setNames(.,c('Low','Medium','High'))          
  
head(prob_pred_scaled)


# MLR: Multinomial Logistic Regression
# MLR_SA: Multinomial Logistic Regression with Sentiment Analysis
# MLR_SD: Multinomial Logistic Regression with Scaled Data
# RF: Random Forest
# RF_SA: Random Forest with Sentiment Analysis

#Comparisons of accuracies:
MLR <- acc_1
MLR_SA <- acc_2
MLR_SD <- acc_scaled
RF <- acc_RF_1
RF_SA <- acc_RF_2

Model <-  c("MLR", "MLR_SA", "MLR_SD", "RF", "RF_SA")
Accuracy  <- c(MLR, MLR_SA, MLR_SD, RF, RF_SA)

ACC <- data.frame(Model = Model, Accuracy  = Accuracy)
ACC
ggplot(ACC, aes(Model, Accuracy, group = 1)) + 
  geom_point(aes(color = Model, size = Accuracy)) + 
  geom_line() + 
  ggtitle("Comparison of accuracies of different models \n on training data")


#Comparisons of run time of models:

Model <-  c("MLR", "MLR_SA", "MLR_SD", "RF", "RF_SA")
RunTime  <- c(RT_MLR, RT_MLR_SA, RT_MLR_SD, RT_RF, RT_RF_SA)

RT <- data.frame(Model = Model, Accuracy  = Accuracy)
RT
ggplot(RT, aes(Model, RunTime, group = 1)) + 
  geom_point(aes(color = Model, size = RunTime)) + 
  geom_line() + 
  ggtitle("Comparison of run time (in sec) of \n different models on training data")


# I did try to perform sentiment analysis but it didn't give any significant improvement in prediction accuracy. The reason behind this, as per my perspective, is: The data set actually comes with description of the apartment listings. It does not come with people's feedback and reviews for the apartments. Description just describes **_How good is the apartment?_** It means that it's not actually people's experience of staying in that apartment, it's just textual presentation of apartment's features. 

# Sentiment Analysis may have produced good results if we were given people's feedbacks and reviews which contains people's feelings about the apartment such as whether the apartment is clean or not, it's safe or not, etc.. This gives different sentiments and then we can analyze them. But unfortunately it's not part of the data set. So, Sentiment Analysis model does not produce promising results.
