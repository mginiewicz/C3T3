# Title: C3T3 - Multiple Regression in R

#Updated:  7/19/2022


###############
# Project Notes
###############
#Volume = DV


# Clear console: CTRL + L


###############
# Housekeeping
###############

# Clear objects if necessary
rm(list = ls())

# get woring directory
getwd()

# set working directory 
setwd("C:/Users/giniewic/OneDrive - HP Inc/Documents/Personal/UT Data Analytics Cert/Course 3/C3T3")
dir()


###############
# Load packages
###############
install.packages("Rtools")
install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("mlbench")
install.packages("doParallel")
install.packages("reshape2")
install.packages("dplyr")
library(caret)
library(corrplot)
library(readr)
library(mlbench)
library(doParallel)
library(e1071)
library(gbm)
library(ggplot2)
library(writexl)
library(reshape2)
library(dplyr)



#####################
# Parallel Processing
#####################

#detectCores()         #detect number of cores
#cl <- makeCluster(2)  # select number of cores
#registerDoParallel(cl) # register cluster
#getDoParWorkers()      # confirm number of cores being used by RStudio
#  Stop Cluster -- After performing tasks, make sure to stop cluster
#stopCluster(cl)
#detectCores()


####################
# Import data
####################

##-- Load training / existing complete dataset (Dataset 1) --##
CompOOB <- read.csv("existingproductattributes2017.csv", stringsAsFactors = FALSE)
str(CompOOB)

##-- Load predict / new incomplete dataset (Dataset 2) --##

IncompOOB <- read.csv("newproductattributes2017.csv", stringsAsFactors = FALSE)
str(IncompOOB)

##-- Load preprocessed datasets that have been saved --##

read.csv("existingproductattributes2017.csv", stringsAsFactors = FALSE)
read.csv("newproductattributes2017.csv", stringsAsFactors = FALSE)


######################
# Save datasets
######################




##################
# Evaluate data
##################

##--- Dataset 1 = Complete OOB ---##

str(CompOOB)

# view first/last obs/rows
head(CompOOB)
tail(CompOOB)
anyNA(CompOOB)
anyDuplicated(CompOOB)
summary(CompOOB)
##### There are null but no duplicates in Complete OOB dataset#####

#Delete attribute with missing data
CompOOB$BestSellersRank <- NULL

# Check again for NA
anyNA(CompOOB)
#### There are no longer any null values in Complete OOB dataset#####

summary(CompOOB)


# Check for outliers in data - may impact accuracy of models

boxplot(CompOOB$Volume, ylab="Volume")

## There are 2 products with volume much higher than others
# Create new datasets removing the outliers (where Volume >5000)

subreadydataOOB <- subset(readyDataOOB, Volume<5000)
summary(subreadydataOOB$Volume)

subreadydata <- subset(readyData, Volume<5000)
summary(subreadydata$Volume)


##--- Dataset 2 = Incomplete OOB ---##

str(IncompOOB)
# view first/last obs/rows
head (IncompOOB)
tail(IncompOOB)
anyNA(IncompOOB)
anyDuplicated(IncompOOB)
##### No null and no duplicates in Incomplete OOB dataset#####

#

#############
# Preprocess
#############


#####################
# EDA/Visualizations
#####################

# Statistics
summary(CompOOB)
summary(IncompOOB)

# Plots
#plot(CompOOB$salary, CompOOB$brand)


################
# Sampling
################

# Create 20% sample of CompOOB data
# Set seed
set.seed(123)
CompOOBSample <- CompOOB[sample(1:nrow(CompOOB), round(nrow(CompOOB)*.2),replace=FALSE),]
nrow(CompOOBSample) # ensure number of obs
head(CompOOBSample) # ensure randomness

# 24 sample - to match number in incomplete dataset
set.seed(123)       # set random seed
CompOOB24 <- CompOOB[sample(1:nrow(CompOOB), 24, replace=FALSE),]
nrow(CompOOB24) # ensure number of obs
head(CompOOB24) # ensure randomness

#######################
# Feature selection
#######################

#######################
# Correlation analysis
#######################

# for regression problems, the below rules apply.
# 1) compare each IV to the DV, if cor > 0.95, remove
# 2) compare each pair of IVs, if cor > 0.90, remove the
#    IV that has the lowest cor to the DV. (see code
#    below for setting a threshold to automatically select
#    IVs that are highly correlated)

# for classification problems, the below rule applies.
# 1) compare each pair of IVs, if cor > 0.90, remove one
#    of the IVs. (see code below to do this programmatically)

# calculate correlation matrix for all vars in dataset except ProductType
# Get column names
colnames(CompOOB)

CorrData <- cor(CompOOB[2:17])
CorrData
corrplot(CorrData, method = "circle", main="Correlation of Whole Dataset")
# do another plot, sorted based on level of collinearity
corrplot(CorrData, order="hclust", main="Correlation of Whole Dataset") 

# find IVs that are highly correlated (ideally >0.90)
corrIV <- cor(CompOOB[2:17])

# create object with indexes of highly correlated features
corrIVhigh <- findCorrelation(corrIV, cutoff=0.9) 

#print indexes of highly correlated attributes
corrIVhigh
colnames(CompOOB[corrIVhigh])
# Highly correlated variables:
## Volume, x5StarReviews
## x4StarReviews, x3StarReviews
## x2StarReviews, x1StarReviews
## x4StarReview is most highly correlated with Volume

# Create new dataset, removing x5StarReviews, x3StarReviews and x1StarReviews
CompOOBCorr = subset(CompOOB, select = -c(x5StarReviews, x3StarReviews, x1StarReviews))
colnames(CompOOBCorr)
head(CompOOBCorr)

# Check correlation of new dataset
colnames(CompOOBCorr)

CorrData2 <- cor(CompOOBCorr[2:14])
CorrData2
corrplot(CorrData2, method = "circle", main="Correlation of Whole Dataset w/ High Correlated Removed")

################
#Transform data#
################

# Dummify the data for OOB dataset
newDataFrameOOB <- dummyVars(" ~ .", data = CompOOB)
readyDataOOB <- data.frame(predict(newDataFrameOOB, newdata = CompOOB))


# Dummify the data for new data with highly correlated removed
newDataFrame <- dummyVars(" ~ .", data = CompOOBCorr)
readyData <- data.frame(predict(newDataFrame, newdata = CompOOBCorr))

# View dataset
str(readyDataOOB)
str(readyData)


##################
# Train/test sets
##################

# lmFuncs - linear model
# rfFuncs - random forests
# nbFuncs - naive Bayes
# treebagFuncs - bagged trees

# For CompOOB - 75% will be for training; 25% for testing 
# Set seed
set.seed(123)
trainSizeOOB <- round(nrow(readyDataOOB)*0.75)
testSizeOOB <- nrow(readyDataOOB)-trainSizeOOB

trainSize <- round(nrow(readyData)*0.75)
testSize <- nrow(readyData)-trainSize

#see how many are in each set
trainSizeOOB
testSizeOOB

trainSize
testSize

#Create training and test sets 
training_indices_OOB <- sample(seq_len(nrow(readyDataOOB)), size=trainSizeOOB)
trainSetOOB <- readyDataOOB[training_indices_OOB,]
testSetOOB <- readyDataOOB[-training_indices_OOB,]

training_indices <- sample(seq_len(nrow(readyData)), size = trainSize)
trainSet <- readyData[training_indices,]
testSet <- readyData[-training_indices,]


# Create training and test sets of new subset datasets (without outliers)
# For CompOOB - 75% will be for training; 25% for testing 
# Set seed
set.seed(123)
subtrainSizeOOB <- round(nrow(subreadydataOOB)*0.75)
subtestSizeOOB <- nrow(subreadydataOOB)-subtrainSizeOOB

subtrainSize <- round(nrow(subreadydata)*0.75)
subtestSize <- nrow(subreadydata)-subtrainSize

#see how many are in each set
subtrainSizeOOB
subtestSizeOOB

subtrainSize
subtestSize

#Create training and test sets 
sub_training_indices_OOB <- sample(seq_len(nrow(subreadydataOOB)), size=subtrainSizeOOB)
subtrainSetOOB <- subreadydataOOB[sub_training_indices_OOB,]
subtestSetOOB <- subreadydataOOB[-sub_training_indices_OOB,]

sub_training_indices <- sample(seq_len(nrow(subreadydata)), size = subtrainSize)
subtrainSet <- subreadydata[sub_training_indices,]
subtestSet <- subreadydata[-sub_training_indices,]




################
# Train control
################

# set cross validation -- splits into 10 for cross-validation
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1, search = 'random')

###############
# Train models
###############


## ---- LINEAR MODEL ---- ##

# X - IV (predictor); Y=DV
# We want to predict volume, so that is our DV=Y; Everything else is IV=X

### OOB dataset ###
modelLookup('lm')
set.seed(123)

OOBLMfit <- train(Volume~., data=trainSetOOB, method = "lm", trControl=fitControl)
OOBLMfit

#RMSE          Rsquared  MAE         
# 4.195232e-13  1         3.154685e-13

##### Rsquared is 1 because Volume and x5StarReviews are perfectly correlated #####

### New dataset - reduced variables ###
set.seed(123)
LMfit <- train(Volume~., data=trainSet, method = "lm", trControl = fitControl)
LMfit

#RMSE      Rsquared   MAE    
#972.5787  0.6335652  729.232


## ---- SUPPORT VECTOR MACHINE (SVM) - OOB Dataset---- ##

modelLookup('svmLinear2')
set.seed(123)

OOBSVMfit <- train(Volume~., data=trainSetOOB, method="svmLinear2", trControl=fitControl)
OOBSVMfit

#cost        RMSE      Rsquared   MAE     
#2.195659  586.5311  0.9090129  305.9756
#303.438653  586.5311  0.9090129  305.9756
#551.420566  586.5311  0.9090129  305.9756

#RMSE was used to select the optimal model using the smallest value.
#The final value used for the model was cost = 2.195659.

varImp(OOBSVMfit)

#only 20 most important variables shown (out of 27)

#Overall
#x5StarReviews              100.0000
#x4StarReviews               94.3069
#PositiveServiceReview       81.5498
#x3StarReviews               71.1142
#x1StarReviews               60.8343
#x2StarReviews               55.6952
#NegativeServiceReview       32.6633
#ProductTypeGameConsole      15.4144
#ProductNum                  13.9289
#ProductWidth                 7.0411
#ProductHeight                6.9205
#ProductDepth                 6.4029
#ShippingWeight               6.3141
#Price                        4.1422
#Recommendproduct             3.2920
#ProfitMargin                 2.3668
#ProductTypePrinter           2.2195
#ProductTypeAccessories       1.0641
#ProductTypePrinterSupplies   0.6542
#ProductTypePC                0.5866

plot(OOBSVMfit, main="Support Vector Machine - OOB")

## ---- SUPPORT VECTOR MACHINE (SVM) - reduced variables ---- ##
set.seed(123)

ReducedSVMfit <- train(Volume~., data=trainSet, method="svmLinear2", trControl=fitControl)
ReducedSVMfit

#cost        RMSE      Rsquared   MAE     
#2.195659   918.973  0.7641499  660.7155
#303.438653  1055.374  0.7331303  756.5140
#551.420566  1056.404  0.7312152  757.5456

varImp(ReducedSVMfit)

#Overall
#x4StarReviews              100.0000
#PositiveServiceReview       90.0604
#x2StarReviews               74.6951
#NegativeServiceReview       25.9232
#ProductNum                  16.6712
#ProductTypeGameConsole      16.2286
#ProductWidth                 7.5165
#ProductHeight                5.0796
#ProfitMargin                 4.3061
#ShippingWeight               4.0456
#Recommendproduct             3.9074
#Price                        3.5204
#ProductDepth                 3.5096
#ProductTypeAccessories       2.2291
#ProductTypePrinter           1.5556
#ProductTypePrinterSupplies   1.2916
#ProductTypeNetbook           0.7758
#ProductTypePC                0.7718
#ProductTypeLaptop            0.4715
#ProductTypeSmartphone        0.3613


## ---- Random Forest (RF) - OOB Dataset ---- ##

# default
set.seed(123)
oobRFfit <- train(Volume~., data=trainSetOOB, method="rf", importance=T,trControl=fitControl)
oobRFfit    

#mtry  RMSE      Rsquared   MAE     
#3    713.4469  0.8991269  377.4139
#14    638.7823  0.9513506  291.0034
#19    633.0636  0.9543163  286.5359

# manual grid
rfGrid <- expand.grid(mtry=c(17,18,19,20,21))  
set.seed(123)

# fit
oobRFfit <- train(Volume~., data=trainSetOOB, method="rf", 
                  importance=T,
                  trControl=fitControl,
                  tuneGrid=rfGrid)
oobRFfit  

#mtry  RMSE      Rsquared   MAE     
#17    625.4733  0.9512738  287.6926
#18    619.8661  0.9541542  281.9824
#19    609.5797  0.9550219  278.9660
#20    604.8405  0.9582993  274.2051
#21    625.8604  0.9503964  285.3195
#### mtry = 20 was best fit (based on RMSE) ####

plot(oobRFfit, main="Random Forest - OOB")
varImp(oobRFfit)
plot(varImp(oobRFfit), main="Random Forest - OOB - Variable Importance")

#Overall
#x5StarReviews                100.00
#PositiveServiceReview         68.80
#x4StarReviews                 48.44
#x2StarReviews                 41.03
#x3StarReviews                 33.68
#ProductTypeGameConsole        30.50
#ProductNum                    29.24
#ProductTypePrinter            27.58
#x1StarReviews                 25.71
#ProductTypeExtendedWarranty   24.48
#Price                         21.93
#NegativeServiceReview         20.66
#ProductTypeSoftware           19.23
#ProductWidth                  19.12
#ProductDepth                  18.20
#ProductHeight                 17.75
#ShippingWeight                15.44
#ProductTypeTablet             15.21
#Recommendproduct              14.38
#ProductTypePC                 13.99

## ---- Random Forest (RF) - reduced variables ---- ##

# default
set.seed(123)
ReducedRFfit <- train(Volume~., data=trainSet, method="rf", importance=T,trControl=fitControl)
ReducedRFfit

#mtry  RMSE      Rsquared   MAE     
#3    790.6101  0.8534490  534.1226
#14    835.3720  0.9177997  452.2877
#19    765.4091  0.9273700  413.9924

# manual grid
rfGrid2 <- expand.grid(mtry=c(17,18,19,20,21))  
set.seed(123)

# fit
ReducedRFfit <- train(Volume~., data=trainSet, method="rf", 
                  importance=T,
                  trControl=fitControl,
                  tuneGrid=rfGrid2)
ReducedRFfit  

#mtry  RMSE      Rsquared   MAE     
#17    773.6044  0.9273028  419.0748
#18    794.5937  0.9259230  430.4609
#19    785.3042  0.9277450  421.5574
#20    786.7969  0.9267056  424.4601
#21    775.6365  0.9278401  421.4527
#### mtr = 17 was best fit (based on RMSE) ####

plot(ReducedRFfit, main="Random Forest - Reduced Variables")
varImp(ReducedRFfit)
plot(varImp(ReducedRFfit), main="Random Forest - Red. Var. - Variable Importance")

#Overall
#PositiveServiceReview       100.000
#x4StarReviews                64.493
#x2StarReviews                41.522
#ProductTypeGameConsole       19.419
#ProductTypePrinter           18.381
#ProductTypeExtendedWarranty  15.096
#ProductNum                   14.844
#ShippingWeight               13.234
#NegativeServiceReview        12.423
#ProductHeight                11.523
#Recommendproduct              9.963
#ProductTypeNetbook            8.122
#ProductTypePrinterSupplies    7.614
#ProductDepth                  7.093
#ProductWidth                  6.470
#ProductTypeTablet             4.639
#ProfitMargin                  4.104
#ProductTypeAccessories        2.416
#ProductTypeDisplay            2.382
#Price                         2.253


## ---- Gradient Boosting (GBM) - OOB Dataset ---- ##
set.seed(123)
oobGBMfit <- train(Volume~., data=trainSetOOB, method="gbm", trControl=fitControl, verbose=FALSE)
oobGBMfit

#shrinkage  interaction.depth  n.minobsinnode  n.trees  RMSE     Rsquared   MAE     
#0.2745122   2                  9              4415     1245.82  0.6094149  839.9348
#0.3313096  10                 18              2045         NaN        NaN       NaN
#0.5741432   6                 23              4702         NaN        NaN       NaN

plot(oobGBMfit, main="Gradient Boosting - OOB")
varImp(oobGBMfit)

#Overall
#x5StarReviews              100.000
#Price                       72.253
#ProductNum                  64.137
#PositiveServiceReview       57.555
#ShippingWeight              50.825
#x1StarReviews               48.232
#x4StarReviews               44.739
#ProfitMargin                36.285
#ProductHeight               34.363
#x2StarReviews               34.204
#NegativeServiceReview       31.351
#ProductWidth                29.876
#x3StarReviews               25.652
#ProductDepth                19.758
#ProductTypeAccessories       4.963
#Recommendproduct             4.267
#ProductTypeSoftware          0.000
#ProductTypePC                0.000
#ProductTypePrinterSupplies   0.000
#ProductTypeTablet            0.000

plot(varImp(oobGBMfit), main="Stochastic Gradient Boosting Variable Importance")


## ---- Gradient Boosting (GBM) - Reduced Variables ---- ##
set.seed(123)
ReducedGBMfit <- train(Volume~., data=trainSet, method="gbm", trControl=fitControl, verbose=FALSE)
ReducedGBMfit

#shrinkage  interaction.depth  n.minobsinnode  n.trees  RMSE    Rsquared   MAE     
#0.2745122   2                  9              4415     1608.7  0.5659341  1162.093
#0.3313096  10                 18              2045        NaN        NaN       NaN
#0.5741432   6                 23              4702        NaN        NaN       NaN

varImp(ReducedGBMfit)

#Overall
#PositiveServiceReview       100.000
#x2StarReviews                65.132
#ProductNum                   51.550
#Price                        50.742
#x4StarReviews                50.466
#NegativeServiceReview        43.428
#ProductWidth                 43.074
#ProductHeight                42.855
#ShippingWeight               41.984
#ProductDepth                 34.804
#ProfitMargin                 22.649
#Recommendproduct              8.904
#ProductTypeAccessories        3.748
#ProductTypeSoftware           0.000
#ProductTypePC                 0.000
#ProductTypePrinterSupplies    0.000
#ProductTypeTablet             0.000
#ProductTypeGameConsole        0.000
#ProductTypeExtendedWarranty   0.000
#ProductTypeSmartphone         0.000


############################
# Predict testSet/validation
############################

## ---- SUPPORT VECTOR MACHINE (SVM) - OOB Dataset---- ##
# predict with SVM
svmPredOOB <- predict(OOBSVMfit, testSetOOB)

# performance measurement
postResample(svmPredOOB, testSetOOB$Volume)

#RMSE    Rsquared         MAE 
#415.2513981   0.8879142 217.6810638

# plot predicted verses actual
plot(svmPredOOB, testSetOOB$Volume, main="Predicted vs. Actual - SVM - OOB")

# look at predicted values
summary(svmPredOOB)

#Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#-148.421    9.668  251.599  612.776 1065.402 3818.258 

svmPredOOB


## ---- SUPPORT VECTOR MACHINE (SVM) - reduced variables ---- ##
# predict with SVM
svmPredReduced <- predict(ReducedSVMfit, testSet)

# performance measurement
postResample(svmPredReduced, testSet$Volume)

#RMSE    Rsquared         MAE 
#724.8561181   0.5635922 491.0581877 

# plot predicted verses actual
plot(svmPredReduced, testSet$Volume, main="Predicted vs. Actual - SVM - Red. Var")

# look at predicted values
summary(svmPredReduced)

#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#-951.7  -295.5   292.1   479.4  1031.3  3231.3 

svmPredReduced

###****SINCE BOTH SVM HAVE NEGATIVE VALUES, I AM NOT GOING TO RE-DO W/O OUTLIERS***####

## ---- Random Forest (RF) - OOB Dataset ---- ##
# predict with RF
rfPredOOB <- predict(oobRFfit, testSetOOB)

# performance measurement
postResample(rfPredOOB, testSetOOB$Volume)

#RMSE    Rsquared         MAE 
#527.8165710   0.8695785 153.8750667 

# plot predicted verses actual
plot(rfPredOOB, testSetOOB$Volume, main="Predicted vs. Actual - RF - OOB")

# look at predicted values
summary(rfPredOOB)

#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#11.68   26.86  218.70  653.70 1028.26 4443.31 

rfPredOOB


## ---- Random Forest (RF) - reduced variables ---- ##
# predict with RF
rfPredReduced <- predict(ReducedRFfit, testSet)

# performance measurement
postResample(rfPredReduced, testSet$Volume)

#RMSE    Rsquared         MAE 
#224.2107179   0.8413526  94.2506933

# plot predicted verses actual
plot(rfPredReduced, testSet$Volume, main="Predicted vs. Actual - RF - Red. Var.")

# look at predicted values
summary(rfPredReduced)

#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#14.08   36.38   76.15  419.56  975.83 1441.42 

rfPredReduced

## ---- Random Forest (RF) - Sub OOB Dataset ---- ##

# default
set.seed(123)
suboobRFfit <- train(Volume~., data=subtrainSetOOB, method="rf", importance=T,trControl=fitControl)
suboobRFfit    

#mtry  RMSE      Rsquared   MAE      
#3    241.7697  0.9103825  164.11878
#14    135.9734  0.9649065   73.86177
#19    126.8275  0.9662844   67.96013

# manual grid
subrfGrid <- expand.grid(mtry=c(19,20,21,22,23))  
set.seed(123)

# fit
suboobRFfit <- train(Volume~., data=subtrainSetOOB, method="rf", 
                     importance=T,
                     trControl=fitControl,
                     tuneGrid=subrfGrid)
suboobRFfit  

#mtry  RMSE      Rsquared   MAE     
#19    119.8615  0.9694289  64.75840
#20    120.4359  0.9705243  65.76515
#21    114.5120  0.9715671  62.24884
#22    116.4033  0.9710741  63.38316
#23    115.5705  0.9712159  62.77576
#### mtry = 21 was best fit (based on RMSE) ####

plot(suboobRFfit, main="Random Forest - OOB (no outliers)")
varImp(suboobRFfit)
plot(varImp(suboobRFfit), main="Random Forest - OOB (no outliers) - Variable Importance")

#Overall
#x5StarReviews               100.000
#PositiveServiceReview        68.336
#x4StarReviews                36.322
#ShippingWeight               19.110
#ProductWidth                 18.955
#x3StarReviews                18.842
#ProductNum                   13.305
#ProfitMargin                 12.799
#ProductTypePrinter           12.432
#ProductDepth                 11.466
#ProductTypeDisplay           11.266
#ProductTypePC                11.266
#ProductTypePrinterSupplies   11.173
#ProductTypeLaptop            11.025
#ProductTypeSoftware          10.930
#ProductTypeTablet            10.533
#x1StarReviews                10.200
#ProductTypeExtendedWarranty   9.879
#Recommendproduct              8.121
#x2StarReviews                 7.950


## ---- Random Forest (RF) - sub dataset reduced variables (no outliers) ---- ##

# default
set.seed(123)
subReducedRFfit <- train(Volume~., data=subtrainSet, method="rf", importance=T,trControl=fitControl)
subReducedRFfit

#mtry  RMSE      Rsquared   MAE     
#3    288.0365  0.9028256  221.0078
#14    231.3850  0.9142689  137.5268
#19    223.9495  0.9145616  133.0160

# manual grid
subrfGrid2 <- expand.grid(mtry=c(16,17,18,19,20))  
set.seed(123)

# fit
subReducedRFfit <- train(Volume~., data=subtrainSet, method="rf", 
                         importance=T,
                         trControl=fitControl,
                         tuneGrid=subrfGrid2)
subReducedRFfit  

#mtry  RMSE      Rsquared   MAE     
#16    227.9659  0.9140866  133.3276
#17    225.4006  0.9161341  132.2628
#18    225.5245  0.9196409  132.8480
#19    219.8538  0.9211712  129.8712
#20    220.5389  0.9143569  128.9088
#### mtr = 19 was best fit (based on RMSE) ####

plot(subReducedRFfit, main="Random Forest - Reduced Variables (no outliers)")
varImp(subReducedRFfit)
plot(varImp(subReducedRFfit), main="Random Forest - Red. Var. - Variable Importance (no outliers)")

#Overall
#PositiveServiceReview       100.000
#x4StarReviews                37.984
#NegativeServiceReview        15.815
#x2StarReviews                14.302
#Price                        12.906
#Recommendproduct             10.338
#ProductWidth                  9.998
#ProductTypePrinter            9.562
#ProductTypeExtendedWarranty   9.276
#ShippingWeight                8.962
#ProductTypePC                 8.530
#ProductHeight                 5.473
#ProductTypeGameConsole        4.442
#ProductTypeNetbook            4.442
#ProductTypeLaptop             4.442
#ProductTypeSoftware           4.129
#ProductTypePrinterSupplies    3.615
#ProductDepth                  3.029
#ProductNum                    1.554
#ProductTypeAccessories        1.538


## ---- Random Forest (RF) - subOOB Dataset (no outliers) ---- ##
# predict with RF
subrfPredOOB <- predict(suboobRFfit, subtestSetOOB)

# performance measurement
postResample(subrfPredOOB, subtestSetOOB$Volume)

#RMSE   Rsquared        MAE 
#96.4662257  0.9659432 43.7829867

# plot predicted verses actual
plot(subrfPredOOB, subtestSetOOB$Volume, main="Predicted vs. Actual - RF - OOB (no outliers)")

# look at predicted values
summary(subrfPredOOB)

#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#11.39   36.40   80.46  406.87  646.68 1510.30 

subrfPredOOB


## ---- Random Forest (RF) - reduced variables ---- ##
# predict with RF
subrfPredReduced <- predict(subReducedRFfit, subtestSet)

# performance measurement
postResample(subrfPredReduced, subtestSet$Volume)

#RMSE   Rsquared        MAE 
#73.6912131  0.9836108 44.1182200

# plot predicted verses actual
plot(subrfPredReduced, subtestSet$Volume, main="Predicted vs. Actual - RF - Red. Var. (no outliers)")

# look at predicted values
summary(subrfPredReduced)

#Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#9.856   62.662  370.905  589.819 1238.780 1424.507 

subrfPredReduced



## ---- Gradient Boosting (GBM) - OOB Dataset ---- ##
# predict with GBM
gbmPredOOB <- predict(oobGBMfit, testSetOOB)

# performance measurement
postResample(gbmPredOOB, testSetOOB$Volume)

#RMSE     Rsquared          MAE 
#1404.3329181    0.5723688  970.9453979 

# plot predicted verses actual
plot(gbmPredOOB, testSetOOB$Volume, main="Predicted vs. Actual - GBM - OOB")

# look at predicted values
summary(gbmPredOOB)

#Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#-1371.99    14.84   405.81  1048.45  1767.55  5608.86  

gbmPredOOB

## ---- Gradient Boosting (GBM) - Reduced Variables ---- ##
# predict with GBM
gbmPredReduced <- predict(ReducedGBMfit, testSet)

# performance measurement
postResample(gbmPredReduced, testSet$Volume)

#RMSE    Rsquared         MAE 
#935.6874378   0.3230367 692.7864340 

# plot predicted verses actual
plot(gbmPredReduced, testSet$Volume, main="Predicted vs. Actual - GBM - Red. Var")

# look at predicted values
summary(gbmPredReduced)

#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#-2368.4  -540.5   576.6   228.0  1096.2  1814.8   

gbmPredReduced

###****SINCE BOTH GBM HAVE NEGATIVE VALUES, I AM NOT GOING TO RE-DO W/O OUTLIERS***####

##################
# Model selection
##################

## SVM and GBM cannot be the best models because they both have negative volumes

## Random Forest - OOB 
#RMSE    Rsquared         MAE 
#527.8165710   0.8695785 153.8750667 

## Random Forest - reduced variables
#RMSE    Rsquared         MAE 
#224.2107179   0.8413526  94.2506933

## RF - OOB dataset w/o outliers
#RMSE   Rsquared        MAE 
#96.4662257  0.9659432 43.7829867

## RF - reduced variable dataset w/o outliers
#RMSE   Rsquared        MAE 
#73.6912131  0.9836108 44.1182200

#--- Random Forest (reduced var) w/o outliers is the best model---#

# Sample #

oobFitSamples <- resamples(list(rfOOB=oobRFfit, rfred=ReducedRFfit, subrfOOB=suboobRFfit, subrfred=subReducedRFfit))
# output summary metrics for tuned models 
summary(oobFitSamples)

#MAE 
#Min.  1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#rfOOB    13.48491 29.40248  46.51464 274.20513 487.97637 1127.1750    0
#rfred    35.90976 74.24520 116.87603 419.07483 295.09550 2519.2261    0
#subrfOOB 10.63069 43.74695  62.89908  62.24884  85.14822  123.6419    0
#subrfred 30.35984 66.53654 132.55951 129.87118 190.08610  233.8472    0

#RMSE 
#             Min.   1st Qu.    Median     Mean  3rd Qu.      Max. NA's
#rfOOB    17.54876  50.72602  76.70905 604.8405 923.1306 2739.3866    0
#rfred    47.38286 115.97861 204.58570 773.6044 635.5404 4453.1591    0
#subrfOOB 18.19925  65.72936 128.21290 114.5120 165.0394  200.0757    0
#subrfred 31.91902 104.67846 247.91097 219.8538 316.5157  364.5441    0

#Rsquared 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#rfOOB    0.7689437 0.9549505 0.9857890 0.9582993 0.9947560 0.9996474    0
#rfred    0.7525218 0.8958974 0.9416804 0.9273028 0.9857407 0.9932903    0
#subrfOOB 0.8998365 0.9612601 0.9810397 0.9715671 0.9910104 0.9989328    0
#subrfred 0.6695124 0.8735644 0.9663935 0.9211712 0.9966488 0.9995617    0



################################################
# Predict new data (Dataset 2 - Incomplete Data)
################################################

#Transform data#

# Dummify the data for IncompOOB dataset
newDataFrameIncompOOB <- dummyVars(" ~ .", data = IncompOOB)
readyDataIncompOOB <- data.frame(predict(newDataFrameIncompOOB, newdata = IncompOOB))

# Using dummified dataset, remove x5StarReviews, x3StarReviews and x1StarReviews
IncompOOBCorr = subset(readyDataIncompOOB, select = -c(x5StarReviews, x3StarReviews, x1StarReviews))
colnames(IncompOOBCorr)
head(IncompOOBCorr)

# predict for new dataset with no values for DV

finalPred <- predict(subReducedRFfit, IncompOOBCorr)
head(finalPred)
finalPred

plot(finalPred, IncompOOB$Variable, main="Prediction, Incomplete Dataset")

print(finalPred)

summary(finalPred)

summary(CompOOB$Volume)

summary(subreadydataOOB$Volume)

summary(IncompOOB$Volume)

# add predictions to new products dataset

predictdf2 <- data.frame(predictions = c(finalPred))

Incompdf2 <- data.frame(IncompOOB)

Incompdf2$Predictions <- finalPred

write.csv(Incompdf2, "C:/Users/giniewic/OneDrive - HP Inc/Documents/Personal/UT Data Analytics Cert/Course 3/C3T3/c3.T3output.csv", row.names=TRUE)


######################
# Organize Predictions 
######################

OrganizedPredictions <- subset(Incompdf2, 
                               ProductType=="PC" | 
                                 ProductType=="Laptop" | 
                                 ProductType=="Netbook" |
                                 ProductType=="Smartphone")

Predictions <- aggregate(Predictions ~ ProductType, data=OrganizedPredictions, FUN = sum)

PredictionsDF <- data.frame(Predictions)

OrganizedOOB <- subset(CompOOB, 
                       ProductType=="PC" | 
                         ProductType=="Laptop" | 
                         ProductType=="Netbook" |
                         ProductType=="Smartphone")

aggregate(Volume ~ ProductType, data=OrganizedOOB, FUN = sum)

Incompdf2$Sales <- Incompdf2$Price * Incompdf2$Predictions

colnames(Incompdf2)

Incompdf3 <- data.frame(Incompdf2)

write.csv(Incompdf3, "C:/Users/giniewic/OneDrive - HP Inc/Documents/Personal/UT Data Analytics Cert/Course 3/C3T3/c3.T3output (with Sales data).csv", row.names=TRUE)

OrganizedPredictions2 <- subset(Incompdf2, 
                               ProductType=="PC" | 
                                 ProductType=="Laptop" | 
                                 ProductType=="Netbook" |
                                 ProductType=="Smartphone")


SalesData <- aggregate(Sales ~ ProductType, data=OrganizedPredictions2, FUN = sum)

barplot(Predictions$Predictions, names=Predictions$ProductType)

