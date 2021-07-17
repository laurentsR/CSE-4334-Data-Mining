# Ryan Laurents - 1000763099
# Data Mining - CSE 4334
# P2 - Classifiers
# 11/15/2020

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

from IPython.display import display
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

#read from the csv file and return a Pandas DataFrame.
nba = pd.read_csv('NBAstats.csv')

# "Position (pos)" is the class attribute we are predicting.
class_column = 'Pos'

#The dataset contains attributes such as player name and team name.
#We know that they are not useful for classification and thus do not
#include them as features.
feature_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', \
    '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', \
    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']

#Pandas DataFrame allows you to select columns.
#We use column selection to split the data into features and class.
nba_feature = nba[feature_columns]
nba_class = nba[class_column]

train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25)

training_accuracy = []
test_accuracy = []

############################################################################################
############################################################################################
## Before selecting a classifier, we want to test out everything sklearn has to offer.    ##
## When we find one with the best accuracy during testing, we will continue the rest of   ##
## the assignment with that classifier. I have kept all classifiers here, commented out,  ##
## so you may see everything that I tried.                                                ##
############################################################################################
############################################################################################

#linearsvm = LinearSVC(dual = False, C = 0.025, max_iter = 100000).fit(train_feature, train_class)
#print("Test set score: {:.3f}".format(linearsvm.score(train_feature, train_class)))
#print("Test set score: {:.3f}".format(linearsvm.score(test_feature, test_class)))

#nb = GaussianNB().fit(train_feature, train_class)
#print("Test set score: {:.3f}".format(nb.score(test_feature, test_class)))

#tree = DecisionTreeClassifier(ccp_alpha = 0.000001)
#tree.fit(train_feature, train_class)
#print("Training set score: {:.3f}".format(tree.score(train_feature, train_class)))
#print("Test set score: {:.3f}".format(tree.score(test_feature, test_class)))

#tree = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#tree.fit(train_feature, train_class)
#print("Training set score: {:.3f}".format(tree.score(train_feature, train_class)))
#print("Test set score: {:.3f}".format(tree.score(test_feature, test_class)))

#knn = KNeighborsClassifier()
#knn.fit(train_feature, train_class)
#prediction = knn.predict(test_feature)
#print("Test set predictions:\n{}".format(prediction))
#print("Test set accuracy: {:.2f}".format(knn.score(test_feature, test_class)))

#neuralNet = MLPClassifier(alpha=1.0)
#neuralNet.fit(train_feature, train_class)
#averageTraining += neuralNet.score(train_feature, train_class)
#averageTest += neuralNet.score(test_feature, test_class)

############################################################################################
############################################################################################
## In general testing, the MLP classifier tends to outperform others. I also feel a bit   ##
## more comfortable as I have worked with nueral nets before in another class. So we will ##
## continue further testing with the MLP classifer.                                       ##
############################################################################################
############################################################################################

# This loop was set up to calculate the average accuracy over 50 iterations. I used it to test
# various parameters within the MLP classifer. I have left in the averages I recieved from the
# various tests below.

#averageTraining = 0;
#averageTest = 0;
#iterations = 50
#for x in range(0,iterations):                                                               #train/test
    #neuralNet = MLPClassifier(activation = 'identity', alpha=1.0, max_iter=10000) AVERAGES: 0.697/0.614
    #neuralNet = MLPClassifier(activation = 'logistic', alpha=1.0, max_iter=10000) AVERAGES: 0.760/0.646
    #neuralNet = MLPClassifier(activation = 'tanh', alpha=1.0, max_iter=10000)     AVERAGES: 0.959/0.573
    #neuralNet = MLPClassifier(activation = 'relu', alpha=1.0, max_iter=10000)     AVERAGES: 0.862/0.597

    # Each of the above was run multiple times to average the volatility
    # We will continue testing with activation = logistic and tanh, the highest test/train averages respectively

    #neuralNet = MLPClassifier(activation = 'logistic', solver = 'lbfgs', alpha = 1.0, tol = 0.01, max_iter=10000)
    #neuralNet = MLPClassifier(activation = 'tanh', solver = 'lbfgs', alpha = 1.0, tol = 0.01, max_iter=10000)

    # Logistic is the better option. We will continue with the logistic activation.

    #neuralNet.fit(train_feature, train_class)
    #averageTraining += neuralNet.score(train_feature, train_class)
    #averageTest += neuralNet.score(test_feature, test_class)


# Using the best version of the MLPClassifier parameters we have found above, we will fit and calculate scores.
neuralNet = MLPClassifier(activation = 'logistic', solver = 'lbfgs', alpha = 1.0, tol = 0.01, max_iter=10000)
neuralNet.fit(train_feature, train_class)

print("Using the MLPClassifier [Multi-Layer Perceptron]")
print("Adjusted Parameters(activation = 'logistic', solver = 'lbfgs', alpha = 1.0, tol = 0.01, max_iter=10000)\n")
print("Calculating the average accuracy of 50 iterations. Please allow 60 seconds to compute..\n")

averageTraining = 0;
averageTest = 0;
iterations = 50
for x in range(0,iterations):
    neuralNet = MLPClassifier(activation = 'logistic', solver = 'lbfgs', alpha = 1.0, tol = 0.01, max_iter=10000)
    neuralNet.fit(train_feature, train_class)
    averageTraining += neuralNet.score(train_feature, train_class)
    averageTest += neuralNet.score(test_feature, test_class)

print("Training set score: {:.3f}".format(averageTraining / iterations))
print("Test set score: {:.3f}\n".format(averageTest / iterations))

# Use predict and crosstab to generate a confusion matrix
prediction = neuralNet.predict(test_feature)
print("\nConfusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

# Use cross_val_score to use StratifiedKFold while also generating the list of averages.
print()
print("Now the same model using the cross_val_score function, which uses StratifiedKFold while also generating the list of scores. (Please allow up to 30 seconds to compute)")
neuralNetStratified = MLPClassifier(activation = 'logistic', solver = 'lbfgs', alpha = 1.0, tol = 0.01, max_iter=10000)
scores = cross_val_score(neuralNetStratified, nba_feature, nba_class, cv = 10)
print(scores)
print("Average accuracy across all 10 folds: ", (sum(scores)/len(scores)))
