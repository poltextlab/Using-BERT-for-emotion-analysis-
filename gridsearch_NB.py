def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB


featuresfinal = pd.read_csv("path_to_corpus_tsv", sep = '\t')
labels = np.load("path_to_corpus_labels", sep = '\t')

# MinMax scaling is applied to the features
scaler = MinMaxScaler()
featuresfinal = scaler.fit_transform(featuresfinal)

# the parameter space is defined below
alpha = [0, 0.25, 0.5, 0.75, 1]
parameters = dict(alpha=alpha)

clasrep = list()
paramlist = list()

# the loop is essentially a robust non-parametric estimation of the mean, given the high volatility of train-test splits
# it fits an optimized logistic regression for all splits
# the loop outputs a list of dictionaries with results and corresponding parameter information to be extracted later
# retainment of true-predicted label pairs is not implemented at this point
for i in range(100):
    train_features, test_features, train_labels, test_labels = train_test_split(featuresfinal, labels, stratify=labels)
    nb = MultinomialNB()
    nbmodel = GridSearchCV(nb, parameters, cv = 3, scoring = 'f1_weighted', n_jobs = -1)
    nbmodel.fit(train_features, train_labels)
    predictions = nbmodel.predict(test_features)
    classifrep = classification_report(test_labels, predictions, output_dict = True, zero_division=0)
    clasrep.append(classifrep)
    paramlist.append(nbmodel.best_params_)
    print("Finished with run!")


# json files preserve the structure well, and are easy to parse
MyFile = open('clasrep_nb.json', 'w')
json.dump(clasrep, MyFile)
MyFile.close()

MyFile = open('param_nb.json', 'w')
json.dump(paramlist, MyFile)
MyFile.close()
