
# coding: utf-8

# In[388]:

from __future__ import division
import numpy as np
import math
import random
import os
import operator
from collections import Counter

def euclidieanLength(x):
    for i in range(x.shape[1]):
        elength = np.sqrt(sum(np.power(x[:,i],2)))
        
        x[:,i] = x[:,i] / elength
    return x


#####Cross Validation of the whole dataset######
def crossValidation(dataSet,k):
    allSet = []
    allIndex = np.asarray(range(len(dataSet)))
    np.random.shuffle(allIndex)

    m = len(dataSet)
    num_valid = int(m * 0.2 * 3 // 4)  ##num_valid = 734
    for i in range(k):
        test_index = allIndex[(m * i) // k:(m * (i + 1)) // k]
        train_index_with_Valid = np.setdiff1d(allIndex, test_index)
        # train_index_with_Valid = np.array([i for i in allIndex if i not in test_index])
        valid_index = train_index_with_Valid[:num_valid]
        # valid_index = np.random.choice(train_index_with_Valid, num_valid, replace=False)
        train_index = np.setdiff1d(train_index_with_Valid, valid_index)

        train_fold = dataSet[train_index]

        valid_fold = dataSet[valid_index]
        test_fold = dataSet[test_index]
        each = {'train_fold': train_fold, 'valid_fold': valid_fold, 'test_fold': test_fold}
        allSet.append(each)
    return allSet

######Three distance measures
def euclideanDistance(a,b):
    a_onlyx = a[:11]
    b_onlyx = b[:11]
    distance = np.linalg.norm(a_onlyx-b_onlyx)
    return distance

def cosinesimiliarity(a,b):
    a_onlyx = a[:11]
    b_onlyx = b[:11]
    distance = sum(np.multiply(a_onlyx,b_onlyx)) / sum(np.power(a_onlyx,2))*sum(np.power(b_onlyx,2))
    return distance

def chisquaredDistance(a,b):
    a_onlyx = a[:11]
    b_onlyx = b[:11]
    distance = sum(np.power(a_onlyx-b_onlyx,2)/(a_onlyx+b_onlyx))
    return distance

#cosinesimiliarity(all_data[0],all_data[1])

def getNeighbors(trainingSet, testpoint, k):
    distances = []
    for x in range(len(trainingSet)):
        ####distance measure change here
        dist = chisquaredDistance(trainingSet[x], testpoint)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    # print neighbors
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedVotes[0][0]


def getResponse2(neighbors):
    classVotes = {}
    responses = []
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        responses.append(response)
    avg_response = sum(responses) / len(responses)
    response = int(round(avg_response))

    return response

def accuracy(trainset, validset,k):
    valid_y = validset[:,-1]
    predict_list = []
    count = 0
    for i in range(validset.shape[0]):
        predict_list.append(getResponse(getNeighbors(trainset, validset[i], k)))
    for i in range(len(predict_list)):
        if predict_list[i] == valid_y[i]:
            count += 1
    acc = count / len(predict_list)
    return acc

def f1score(trainset, validset,k):
    valid_y = validset[:,-1]
    unique = np.unique(valid_y)
    f1score_list = []
    predict = []
    for i in range(validset.shape[0]):
        predict.append(getResponse(getNeighbors(trainset, validset[i], k)))
    predict = np.asarray(predict)
    for j in unique:
        TP = np.count_nonzero((valid_y == j) & (predict == j))
        FP = np.count_nonzero((valid_y != j) & (predict == j))
        FN = np.count_nonzero((valid_y == j) & (predict != j))
        f1 = 2*TP/(2*TP+FP+FN)
        f1score_list.append(f1)
    return sum(f1score_list)/len(f1score_list)

def main():
    #os.chdir('C:/Users/Henry/Desktop/cs578/hw1')
    my_data = np.genfromtxt('winequality-white.csv', delimiter=';')
    my_data = my_data[1:]
    y = my_data[:, 11].astype(int)
    X = my_data[:, :11]
    X_normed = euclidieanLength(X)
    y = y.reshape(len(y), 1)
    all_data = np.hstack((X_normed, y))
    fold1, fold2, fold3, fold4 = crossValidation(all_data, 4)
    print 'Hyper-paremeters:'
    print 'K:14'
    print 'Distance measure: Euclidean Distance'
    print '  '
    ###fold-1
    fold1_valid_acc = accuracy(fold1['train_fold'], fold1['valid_fold'], 14) * 100 ##change the k of knn here
    fold1_valid_f1 = f1score(fold1['train_fold'], fold1['valid_fold'], 14) * 100
    fold1_test_acc = accuracy(fold1['train_fold'], fold1['test_fold'], 14) * 100
    fold1_test_f1 = f1score(fold1['train_fold'], fold1['test_fold'], 14) * 100
    print 'Fold-1:'
    print 'Validation: F1 score: %s, ' % fold1_valid_f1 + 'Accuracy: %s1' % fold1_valid_acc
    print 'Test: F1 score: %s, ' % fold1_test_f1 + 'Accuracy: %s1' % fold1_test_acc
    print '  '
    ###fold-2
    fold2_valid_acc = accuracy(fold2['train_fold'], fold2['valid_fold'], 14) * 100 ##change the k of knn here
    fold2_valid_f1 = f1score(fold2['train_fold'], fold2['valid_fold'], 14) * 100
    fold2_test_acc = accuracy(fold2['train_fold'], fold2['test_fold'], 14) * 100
    fold2_test_f1 = f1score(fold2['train_fold'], fold2['test_fold'], 14) * 100
    print 'Fold-2:'
    print 'Validation: F1 score: %s, ' % fold2_valid_f1 + 'Accuracy: %s1' % fold2_valid_acc
    print 'Test: F1 score: %s, ' % fold2_test_f1 + 'Accuracy: %s1' % fold2_test_acc
    print '  '
    ###fold-3
    fold3_valid_acc = accuracy(fold3['train_fold'], fold3['valid_fold'], 14)  * 100##change the k of knn here
    fold3_valid_f1 = f1score(fold3['train_fold'], fold3['valid_fold'], 14)* 100
    fold3_test_acc = accuracy(fold3['train_fold'], fold3['test_fold'], 14)* 100
    fold3_test_f1 = f1score(fold3['train_fold'], fold3['test_fold'], 14)* 100
    print 'Fold-3:'
    print 'Validation: F1 score: %s, ' % fold3_valid_f1 + 'Accuracy: %s1' % fold3_valid_acc
    print 'Test: F1 score: %s, ' % fold3_test_f1 + 'Accuracy: %s1' % fold3_test_acc
    print '  '
    ###fold-4
    fold4_valid_acc = accuracy(fold4['train_fold'], fold4['valid_fold'], 1)* 100  ##change the k of knn here
    fold4_valid_f1 = f1score(fold4['train_fold'], fold4['valid_fold'], 1)* 100
    fold4_test_acc = accuracy(fold4['train_fold'], fold4['test_fold'], 1)* 100
    fold4_test_f1 = f1score(fold4['train_fold'], fold4['test_fold'], 1)* 100
    print 'Fold-4:'
    print 'Validation: F1 score: %s, ' % fold4_valid_f1 + 'Accuracy: %s1' % fold4_valid_acc
    print 'Test: F1 score: %s, ' % fold4_test_f1 + 'Accuracy: %s1' % fold4_test_acc
    print '  '
    ###Average
    avg_valid_acc = (fold1_valid_acc + fold2_valid_acc + fold3_valid_acc + fold4_valid_acc) / 4
    avg_valid_f1 = (fold1_valid_f1 + fold2_valid_f1 + fold3_valid_f1 + fold4_valid_f1) / 4
    avg_test_acc = (fold1_test_acc + fold2_test_acc + fold3_test_acc + fold4_test_acc) / 4
    avg_test_f1 = (fold1_test_f1 + fold2_test_f1 + fold3_test_f1 + fold4_test_f1) / 4
    print 'Average:'
    print 'Validation: F1 score: %s, ' % avg_valid_f1 + 'Accuracy: %s1' % avg_valid_acc
    print 'Test: F1 score: %s, ' % avg_test_f1 + 'Accuracy: %s1' % avg_test_acc


main()

