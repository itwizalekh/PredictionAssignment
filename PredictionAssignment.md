Prediction Assignment Writeup
=============================

We Load the Caret Library, and set seed.  


```r
library(caret); library(MASS)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(12345)
```

### Reading Data

Now we read input and test data:  


```r
setwd("C:/Misc/R_work/data")
pmlRaw <- read.csv("pml-training.csv")
Test <- read.csv("pml-testing.csv")
```

### Feature selection

A small random sample of input data set was manually examined for feature selection.
Now, picking only the features of interest, discarding the rest.  

```r
cols<-names(pmlRaw) %in% c("user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","num_window", "roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt", "gyrox_belt_x", "gyrox_belt_y", "gyrox_belt_z", "accel_belt_x", "accel_belt_y", "accel_belt_z", "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", "roll_arm", "pitch_arm", "yaw_arm", "total_acel_arm", "gyros_arm_x", "gyros_arm_y", "gyros_arm_z", "accel_arm_x", "accel_arm_y", "accel_arm_z", "magnet_arm_x", "magnet_arm_y", "magnet_arm_z", "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "total_accel_dumbbell", "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z", "accell_dumbbell_x", "accel_dumbbell_y", "accel_dumbbell_z", "magnet_dumbbell_x", "magnet_dumbbell_y", "magnet_dumbbell_z", "roll_forearm", "pitch_forearm", "yaw_forearm", "total_accel_forearm", "gyros_forearm_y", "gyros_forearm_z", "accel_forearm_x", "accel_forearm_y", "accel_forearm_z", "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z")

pmlSlim <- pmlRaw[,cols]
pmlSlim$classe <- pmlRaw$classe
```

### Partitioning data into Training set and Test set

We set aside 70% of randomly sampled data for training set, and remaining 30% data for use as Test set.  


```r
inTrain <- createDataPartition(y=pmlSlim$classe, p=0.7, list=FALSE)
pmlTrain <- pmlSlim[inTrain,]
pmlTest <- pmlSlim[-inTrain,]
```

### Handling missing values

Here we look for NAs to see if we need to imputed dataset using k-nearest neighbors method.

```r
table(complete.cases(pmlSlim))
```

```
## 
##  TRUE 
## 19622
```

As we can see, our selected features have no NAs, so data imputation is not required.

### Training prediction model

We will now train prediction model using Linear Discriminant Analysis method.


```r
modlda <- train(classe ~ ., method="lda", data=pmlSlim)
modlda
```

```
## Linear Discriminant Analysis 
## 
## 19622 samples
##    51 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
## 
## Resampling results
## 
##   Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.9       0.8    0.004        0.006   
## 
## 
```

We now validate our model by looking at its accuracy in classifying Test Set that we set aside.

```r
pred <- as.vector(predict(modlda,pmlTest))
confusionMatrix(pred,pmlTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1512  149    1    0    0
##          B  145  835  102    1    0
##          C   17  149  885   99    2
##          D    0    6   36  800  103
##          E    0    0    2   64  977
## 
## Overall Statistics
##                                        
##                Accuracy : 0.851        
##                  95% CI : (0.842, 0.86)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.812        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.903    0.733    0.863    0.830    0.903
## Specificity             0.964    0.948    0.945    0.971    0.986
## Pos Pred Value          0.910    0.771    0.768    0.847    0.937
## Neg Pred Value          0.962    0.937    0.970    0.967    0.978
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.257    0.142    0.150    0.136    0.166
## Detection Prevalence    0.282    0.184    0.196    0.161    0.177
## Balanced Accuracy       0.934    0.840    0.904    0.900    0.945
```

Looking at the Confusion Matrix, we note that Accuracy is >85%, and both sensitifity and specificity are >90%.

We now use our model to classify Test data.


```r
predTest <- as.vector(predict(modlda,Test))
results <- data.frame(cbind(Test[,160],predTest))
colnames(results) <- c("problem_id","classe")
results
```

```
##    problem_id classe
## 1           1      B
## 2           2      B
## 3           3      B
## 4           4      A
## 5           5      A
## 6           6      E
## 7           7      D
## 8           8      C
## 9           9      A
## 10         10      A
## 11         11      B
## 12         12      C
## 13         13      B
## 14         14      A
## 15         15      E
## 16         16      E
## 17         17      A
## 18         18      B
## 19         19      B
## 20         20      B
```

We now write our results to output files using the script provided with the assignment instructions:


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predTest)
```
