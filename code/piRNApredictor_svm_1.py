# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:55:40 2016

@author: Nackel
"""

import numpy as np
from numpy import array
from pandas import DataFrame
from itertools import combinations_with_replacement, permutations
import time

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import matthews_corrcoef



# input sequences..............................................................................
def getSequences(f):
    seqslst = []
    while True:
        s = f.readline()
        if not s:
            break
        else:
            if '>' not in s:
                seq = s.split('\n')[0]
                seqslst.append(seq)
    return seqslst


# getting k-mer profile..............................................................
def getKmerMatrix(instances, piRNAletter, k):
    p = len(piRNAletter)
    kmerdict = getKmerDict(piRNAletter, k)
    features = []
    for sequence in instances:
        vector = getKmerVector(sequence, kmerdict, p, k)
        features.append(vector)
    return array(features)


def getKmerDict(piRNAletter, k):
    kmerlst = []
    partkmers = list(combinations_with_replacement(piRNAletter, k))
    for element in partkmers:
        elelst = set(permutations(element, k))
        strlst = [''.join(ele) for ele in elelst]
        kmerlst += strlst
    kmerlst = np.sort(kmerlst)
    kmerdict = {kmerlst[i]: i for i in range(len(kmerlst))}
    return kmerdict


def getKmerVector(sequence, kmerdict, p, k):
    vector = np.zeros((1, p ** k))
    n = len(sequence)
    for i in range(n - k + 1):
        subsequence = sequence[i:i + k]
        position = kmerdict.get(subsequence)
        vector[0, position] += 1
    return list(vector[0])


# prediction based on Kmer.....................................................

def getCrossValidation(X, y, clf, folds):
    '''TODO:SN SP ACC MCC '''
    predicted_probability = -np.ones(len(y))
    predicted_label = -np.ones(len(y))
    X = np.array(X)
    y = np.array(y)
    for train_index, test_index in folds:
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        probability_test = (clf.fit(X_train, y_train)).predict_proba(X_test)
        predicted_probability[test_index] = probability_test[:, 1]
        predicted_label[test_index] = (clf.fit(X_train, y_train)).predict(X_test)

    fpr, tpr, thresholds = roc_curve(y, predicted_probability, pos_label=1)
    auc_score = auc(fpr, tpr)
    accuracy = accuracy_score(y, predicted_label)
    sensitivity = recall_score(y, predicted_label)
    specificity = (accuracy * len(y) - sensitivity * sum(y)) / (len(y) - sum(y))
    MCC = matthews_corrcoef(y, predicted_label)
    return auc_score, accuracy, sensitivity, specificity, MCC


##############################################################################################


if __name__ == '__main__':

    featurename = 'Kmer'
    piRNAletter = ['A', 'C', 'G', 'U']

    # input sequences
    fp_nonfunc = open("../data/piR_piRBank_mouse_nonfunc_709.txt", 'r')  # to be opt
    fp_func = open("../data/piR_piRBase_mouse_func_709.txt")
    #posis = getSequences(fp_func) 
    posis = getSequences(fp_func) + getSequences(fp_nonfunc)
    fn = open("../data/pseudopir_noncode[V3.0]_random_1418.txt", 'r') 
    negas = getSequences(fn)
    #negas = getSequences(fp_nonfunc)
    instances = array(posis + negas)
    y = array([1] * len(posis) + [0] * len(negas))
    print('The number of positive and negative samples: %d, %d' % (len(posis), len(negas)))

    # getting k-mer profiles for k=1,2,3,4,5
    for k in range(1, 6):
        print('...............................................................................')
        print('Coding for ' + str(k) + '-' + featurename + ', beginning')
        tic = time.clock()

        X = getKmerMatrix(instances, piRNAletter, k)
        print('Dimension of ' + str(k) + '-' + featurename + ': %d' % len(X[0]))

        toc = time.clock()
        print('Coding time: %.3f minutes' % ((toc - tic) / 60.0))
        if k == 1:
            all_X = X
        else:
            all_X = np.hstack((all_X, X))
        print('...............................................................................')

    # output the spectrum profile
    np.savetxt(featurename + 'Feature.txt', all_X)
    # prediction based on spectrum profile
    print('The prediction based on ' + featurename + ', beginning')
    tic = time.clock()

#    clf = RandomForestClassifier(random_state=1,n_estimators=500)
    C_range = np.logspace(-5, 15, 11, base = 2)
    gamma_range = np.logspace(3, -15, 10, base = 2)
    param_grid = dict(gamma=gamma_range, C=C_range, kernel = ['rbf'])
    clf = GridSearchCV(svm.SVC(probability=True), param_grid, cv = 5)
    clf.fit(all_X,y)
    print("The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))
    clf_best = clf.best_estimator_
    folds = KFold(len(y), n_folds=5, shuffle=True, random_state=np.random.RandomState(1))
    auc_score, accuracy, sensitivity, specificity, MCC = getCrossValidation(all_X, y, clf, folds)

    print('results for feature:' + featurename)
    print('AUC score:%.3f, ACC:%.3f, SN:%.3f, SP:%.3f， MCC:%.3f' % (auc_score, accuracy, sensitivity, specificity, MCC))
#
#    toc = time.clock()
#    print('The prediction time: %.3f minutes' % ((toc - tic) / 60.0))
    # print('###############################################################################\n')
    #
    # # output result
    # results = DataFrame({'Feature': [featurename], \
    #                      'AUC': [auc_score], \
    #                      'ACC': [accuracy], \
    #                      'SN': [sensitivity], \
    #                      'SP': [specificity]})
    # results = results[['Feature', 'AUC', 'ACC', 'SN', 'SP']]
    # results.to_csv(featurename + 'Results.csv', index=False)
