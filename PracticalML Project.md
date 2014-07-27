Practical Machine Learning Project
========================================================

## Executive Summary

This file contains the Project's write-up of the _Practcal ML_ course.
The main purpose of this analysis is to find a predictor  model which can accurately classifies a set of movements among 5 categories. This analysis is based on the paper _Qualitative Activity Recognition of Weight Lifting Exercises_ in which given a dataset the investigation was focused on how well an activity was performed by an individual, in particular, the Unilateral Dumbbell Biceps Curl activity.

## Data description

The data was extracted from _https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv_ This data provides different measurements (variables) from devices of how the exercise was executed for 6 different individuals. There are 5 categories which define how well the activity was performed: (Class A) exactly according to the specification , (Class B) throwing the elbows to the front , (Class C) lifting the dumbbell only halfway , (Class D) lowering the dumbbell only halfway  and (Class E) throwing the hips to the front .

Additionally a set of testing data was extracted from _https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv_ which is being used to predict how well an individual performed the exercises using the prediction algorithm which will be described in the next sections.

## Cleaning and Splitting the Data

The training set needed to be cleaned because there were certain variables with missing values for most of the observations and/or were NA's. Additionally the first column of the training data was just the index of the row but does not bring any added value, it rather introduces unnecessary variance.

I cleaned the training and testing set mostly using Excel so that when reading it on R was already NAs free or empty values:



```r
#Setting working directory and necessary libraries
setwd("C:/Users/gabriela_delacerda/Gabriela-Files/Persona/Cursera/Practical ML/Project")
```

```
## Error: cannot change working directory
```

```r
library(lattice)
library(ggplot2)
library(caret)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
#reading the training data
data<-read.csv("pml-training wo NAs.csv")
```

I split my data 50% training and 50% for testing. I did this because the training set was an already large data set and using the entire set would take a while to run  and is a good starting point, on a second try if necessary more data could be chosen as training. 


```r
inTrain<-createDataPartition(y=data$classe,p=0.5,list=FALSE)
testing<-data[-inTrain,]
training<-data[inTrain,]
```

Do not confuse this testing data with the data from pml-testing.csv file, I refer to the data from this file as the assignment problems.
On a first try of the analysis I did not removed the first column from the data, the one which looks like just indices of the rows. And this lead me to a poor predictor model, as I got 7 incorrect answers from the 20 problems we had to predict. This made me revise the data and have a closer look, thus I realized that this "index" column was rather causing extra variance and making the training to run slower.
 

```r
training<-training[,2:60]
training<-training[,!(colnames(training) %in% c("user_name","cvtd_timestamp","new_window"))]
testing<-testing[,2:60]
testing<-testing[,!(colnames(testing) %in% c("user_name","cvtd_timestamp","new_window"))]
```

##Preprocessing and Training

I wanted to run a PCA to cut the large number of variables and just use the transformation of it which will provide the most representative portion of the data and choosing to retain the 99% of the variance.
I planned to use to train the model using the random forest method because this is a classification problem and random forest a known for being fairly accurate.


```r
#remove the col 56 which contains the classe variable
preProcTrain<-preProcess(training[,-56],method="pca",thresh=0.99)
trainPC<-predict(preProcTrain,training[,-56])
modelFit<-train(training$classe~.,method="rf",data=trainPC)
```

The model results are shown in the Appendix. It is more intuitive to interpret the accuracy of the model seen with a confusion matrix using the ttesting set from the split, and this is what I've done next.



##Cross Validation

Before attempting to predict the testing data for the assignment, I used my testing data from the split to check the performance of mu model


```r
 testing<-testing[,!(colnames(testing) %in% c("user_name","cvtd_timestamp","new_window"))]
 testPC<-predict(preProcTrain,testing[,-56])
 cm<-confusionMatrix(testing$classe,predict(modelFit,testPC))
```
 
 In this confusion matrix we can see that the larger numbers are concentrated on the diagonal which is what is expected to happen and few small numbers off the diagonal. The accuracy is 97% which provided me with good confidence that the model does well and will predict the assignment problems with great accuracy.
 
 
 ## Assignment Problems

Similar to the training data, the data provided to make new predictions from the pml-testing.csv file was cleaned in the same way as I did for the training.



```r
problems<-read.csv("pml-testing wo NAs.csv")
problems<-problems[,2:59]
problems<-problems[,!(colnames(problems) %in% c("user_name","cvtd_timestamp","new_window"))]
problemsPC<-predict(preProcTrain,problems)
```
 
The results are shown in the Appendix. Just one out of the 20 predictions was incorrect, presumably due to over-fitting.




Appendix
========================================================

Random Forest 

9812 samples
  38 predictors
   5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Bootstrapped (25 reps) 

Summary of sample sizes: 9812, 9812, 9812, 9812, 9812, 9812, ... 

Resampling results across tuning parameters:

  mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
  2     0.956     0.944  0.00472      0.00599 
  20    0.943     0.928  0.00482      0.00614 
  39    0.93      0.911  0.00768      0.00976 

Accuracy was used to select the optimal model using  the largest value.
The final value used for the model was mtry = 2. 



Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 2777    3    3    3    4
         B   50 1805   41    1    1
         C    0   32 1662   12    5
         D   12    6   77 1511    2
         E    1    8   19   13 1762

Overall Statistics
                                          
               Accuracy : 0.9701          
                 95% CI : (0.9666, 0.9734)
    No Information Rate : 0.2895          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9622          
 Mcnemar's Test P-Value : < 2.2e-16       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9778   0.9736   0.9223   0.9812   0.9932
Specificity            0.9981   0.9883   0.9939   0.9883   0.9949
Pos Pred Value         0.9953   0.9510   0.9714   0.9397   0.9773
Neg Pred Value         0.9910   0.9938   0.9827   0.9965   0.9985
Prevalence             0.2895   0.1890   0.1837   0.1570   0.1808
Detection Rate         0.2831   0.1840   0.1694   0.1540   0.1796
Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
Balanced Accuracy      0.9880   0.9809   0.9581   0.9847   0.9941


classePrediction
 [1] B A C A A E D B A A B C B A E E A B B B
Levels: A B C D E
