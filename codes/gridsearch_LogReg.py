def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


featuresfinal = pd.read_csv("corpora_and_labels/v10_corpus.tsv", sep = '\t')
labels = pd.read_csv("corpora_and_labels/v10_labels.tsv", sep = '\t')

# MinMax scaling is applied to the features
scaler = MinMaxScaler()
featuresfinal = scaler.fit_transform(featuresfinal)

# the parameter space is defined below
C = [0.1, 1]
tol = [0.001, 0.005, 0.01]
weighting = ['balanced']
solver = ['liblinear']
max_iter = [8000]
parameters = dict(C=C, tol=tol, class_weight=weighting, solver=solver, max_iter=max_iter)

clasrep = list()
paramlist = list()

# the loop is essentially a robust estimation of the mean, given the high volatility of train-test splits
# it fits an optimized logistic regression for all splits
# the loop outputs a list of dictionaries with results and corresponding parameter information to be extracted later
# retainment of true-predicted label pairs is not implemented at this point
for i in range(3):
    train_features, test_features, train_labels, test_labels = train_test_split(featuresfinal, labels, stratify=labels)
    lr = LogisticRegression()
    lrmodel = GridSearchCV(lr, parameters, cv = 3, scoring = 'f1_weighted', n_jobs = -1)
    lrmodel.fit(train_features, train_labels)
    predictions = lrmodel.predict(test_features)
    classifrep = classification_report(test_labels, predictions, output_dict = True)
    clasrep.append(classifrep)
    paramlist.append(lrmodel.best_params_)
    print("Finished with run!")


# json files preserve the structure well, and are easy to parse
MyFile = open('clasrep_lr.json', 'w')
json.dump(clasrep, MyFile)
MyFile.close()

MyFile = open('param_lr.json', 'w')
json.dump(paramlist, MyFile)
MyFile.close()
