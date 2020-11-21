# Credit Risk Analysis

## Overview of the analysis: 
The objective of this analysis is to determine which machine learning models best represent credit risk for LendingClub loan applicants. Since the provided data contains a significant imbalance between the quantities of good (low-risk) loans and bad (high-risk) loans, we ran several algorithms to resample the data and balance out the high-risk loan data so that machine learning models do not contain bias towards the larger (majority) low-risk loan data. The models we ran  oversample and under sample the dataset, as well employing a combinatorial model that both over -and- under samples data and finally we used two models that reduce bias in sample data, in an effort to find an optimal model.

## Resources
* Python 3
* Jupyter Notebook
* VS Code
* Machine Leaning libraries:
    * imblearn
    * scikit learn


## Results: 

* The Naive Random Oversampling method increases the size of the minority class (high risk loans) by randomly selecting and adding instances from this same dataset. Accuracy and recall scores for this model are above 0.6 however, the precision and F1 scores for the high risk class are both very low (0.01 and 0.02, respectively)
    

![NaiveRandomOversampling](./additional_resources/NaiveRandomOversampling.png) 
Random Oversampling 
 * balanced accuracy score: 0.67 
    * precision scores 
        * high risk 0.01
        * low risk 1
    * recall scores
        * high risk 0.74
        * low risk 0.61
    * F1 Score
        * high risk 0.02

* The SMOTE algorithm oversamples the high risk loan (minority) class by creating new instances that are interpolated from the neearest neighbors of an existing instance, instead of being randomly selected. Running this model on the LendingClub data produces a slightly lower accuracy rate and similarly very low high risk precision and F1 scores.

![SMOTE](./additional_resources/SMOTE.png) 
* SMOTE Algorithm overSample
    
    * balanced accuracy score: 0.66 
    * precision  
        * high risk 0.01
        * low risk 1

    * recall scores
        * high risk 0.63
        * low risk 0.69
    * F1 Score
        * high risk 0.02

* The Cluster Centroids method undersamples the majority class, in our case low risk loans, to rebalance the proportion of low risk to high risk loans in our testing dataset. In this iteration, this model has almost exactly the same accuracy and precision as well as recall for high risk loans.
 
![ClusterCentroids](./additional_resources/ClusterCentroids.png) 
* ClusterCentroids undersample
    * balanced accuracy score: 0.67
    * precision  
        * high risk 0.01
        * low risk 1

    * recall scores
        * high risk 0.63
        * low risk 0.69
    * F1 Score
        * high risk 0.02

* The SMOTEENN algorithm combines oversampling with undersampling to rebalance the loan classes. 
![SMOTEENN](./additional_resources/SMOTEENN.png) 
* SMOTEENN algorithm
    * balanced accuracy scores 
    * precision  
        * high risk
        * low risk

    * recall scores
        * high risk
        * low risk
    * F1 Score
        * high risk 0.02

* BalancedRandomForestClassifier two new machine learning models that reduce bias, 
    * balanced accuracy score: 0.78
    * precision  
        * high risk 0.03
        * low risk 1.00

    * recall scores
        * high risk 0.70
        * low risk 0.87
    * F1 Score
        * high risk 0.06

![BalancedRandomForestClassifier](./additional_resources/BalancedRandomForestClassifier.png)

* EasyEnsembleClassifier
    * balanced accuracy score: 0.93
    * precision  
        * high risk 0.09
        * low risk 1.00

    * recall scores
        * high risk 0.92
        * low risk 0.94
    * F1 Score
        * high risk 0.16


![EasyEnsembleClassifier](./additional_resources/EasyEnsembleClassifier.png) 

## Summary: 
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. 
* There is a summary of the results (2 pt)
* If you do not recommend any of the models, justify your reasoning.
* There is a bulleted list that describes the balanced accuracy score and the precision and recall scores of all six machine learning models (15 pt)


There is a recommendation on which model to use, or there is no recommendation with a justification (3 pt)