
# coding: utf-8

from __future__ import division
import numpy as np
import math
import os
import random

#####Cross Validation of the whole dataset######
def crossValidation(dataSet,k):

    allSet = []
    allIndex = np.asarray(range(len(dataSet)))
    np.random.shuffle(allIndex)

    m = len(dataSet) ##m=4898
    num_valid = int(m*0.2*3//4)##num_valid = 734
    for i in range(k):
        test_index = allIndex[(m*i)//k:(m*(i+1))//k] 
        train_index_with_Valid = np.setdiff1d(allIndex, test_index)
        #train_index_with_Valid = np.array([i for i in allIndex if i not in test_index])
        valid_index = train_index_with_Valid[:num_valid]
        #valid_index = np.random.choice(train_index_with_Valid, num_valid, replace=False)
        train_index = np.setdiff1d(train_index_with_Valid, valid_index)

               
        train_fold =dataSet[train_index]
        train_fold = train_fold[train_fold[:,-1].argsort()] 
        valid_fold = dataSet[valid_index]
        valid_fold = valid_fold[valid_fold[:,-1].argsort()]
        test_fold = dataSet[test_index]
        test_fold = test_fold[test_fold[:,-1].argsort()]
        each ={'train_fold':train_fold,'valid_fold':valid_fold,'test_fold':test_fold}
        allSet.append(each)
    return allSet#len(eachfold_test_index)
# train1 = crossValidation(data,4)[1]['train_fold']
# valid1 = crossValidation(data,4)[1]['valid_fold']
# train2 = crossValidation(data,4)[2]['train_fold']
# valid2 = crossValidation(data,4)[2]['valid_fold']
# print train1
# print valid1

def entropy2(labels):
    items = np.unique(labels)
    if items.size == 1:
        return 0
    counts = np.zeros((items.shape[0], 1))
    sums = 0
    for x in range(items.shape[0]):
        counts[x] = sum(labels == items[x]) / (labels.size * 1.0)
    for count in counts:
        sums += -1 * count * math.log(count, 2)
    return sums

def getThreshold2(dataset):
    #print dataset
    labels = dataset[:,-1]
    X = dataset[:,:-1]
    gains = np.zeros((labels.shape[0], 1))
    baseEntroy = entropy2(labels)
    thresholdEachAttr = np.zeros((X.shape[1],1))
    for i in range(labels.shape[0]):
        subarray_left = labels[:i]
        subarray_right = labels[i:]
        gains[i] = baseEntroy - subarray_left.shape[0]/labels.shape[0]*entropy2(subarray_left) - subarray_right.shape[0]/labels.shape[0]*entropy2(subarray_right)
    splitvalue_index = np.argmax(gains)
    
    splitvalue_y =labels[splitvalue_index]
    for m in range(X.shape[1]):
        thresholdEachAttr[m] = X[:,m][splitvalue_index]
    return splitvalue_index,splitvalue_y, thresholdEachAttr
        
# def getThreshold(X,y,uniqueSet): ##will get [[.99],[0.22],[],...] 11 个
#     thresholds = np.zeros((X.shape[1], 1))
    
#     for i in range(X.shape[1]):
    
#         newSet = np.hstack((X[:,i].reshape(len(X[:,0]),1), y.reshape(len(y),1)))
#         #print newSet
        
#         #print newSet[:,0][:-1]
#         thresholdSet = np.diff(newSet[:,0])/2+newSet[:,0][:-1]
#         #print thresholdSet
#         errors = np.zeros((thresholdSet.shape[0], 1))
#         for x in range(thresholdSet.shape[0]):
#             ###change here以前min(uniqueSet)+max(uniqueSet))/2是6
#             errors[x] = (newSet[:,1][:x]> (min(uniqueSet)+max(uniqueSet))/2).sum() + (newSet[:,1][x:]<= (min(uniqueSet)+max(uniqueSet))/2).sum()
        
        
#           #print errors
#         thresholds[i] = thresholdSet[np.argmax(errors)]
#     return thresholds
#getThreshold2(data)

#a = np.asarray([1,2,2,2,2,4,5,6,7,7,7,7,8])
#a[:3]

def split(dataset,attribute):
    #labels = dataset[:,-1]
    #X = dataset[:,:-1]
    #print "split function attribute value:   "+ str(attribute)
    #print "based on the attribute up, we got the threshold of that attribute for split of :  " + str(getThreshold(dataset[:,:-1],dataset[:,-1],uniqueSet)[attribute][0])
    subsetLeft = dataset[dataset[:,attribute] <= getThreshold2(dataset)[2][attribute][0]]
    subsetRight = dataset[dataset[:,attribute] > getThreshold2(dataset)[2][attribute][0]]
    return subsetLeft, subsetRight
#split(data,10)

def bestFeature(dataset):
    X = dataset[:,:-1]
    #print "bluuu" + str(X.shape[1]-1)
    labels = dataset[:,-1]
    baseEntropy = entropy2(labels)
    inforgains = np.zeros((X.shape[1],1))
    for eachAttr in range(X.shape[1]):
        #print eachAttr
        left, right = split(dataset,eachAttr)
        labels_left = left[:,-1]
        labels_right = right[:,-1]
        en_left = entropy2(labels_left)
        en_right = entropy2(labels_right)
        inforgains[eachAttr] = baseEntropy - en_left*left.shape[0]/dataset.shape[0] - en_right*right.shape[0]/dataset.shape[0]
    bestattr = np.argmax([inforgains])
    return bestattr

def getSplit(dataset):
    attr = bestFeature(dataset)
    #print "the attr in getSplit function" + str(attr)
    
    left, right = split(dataset,attr)
    return {'attribute': attr, 'left': left, 'right': right}

def terminalMajorityVote(labels):
    #print labels
    labels = labels.astype(int)
    return np.argmax(np.bincount(labels))

def treesplit(node, depth, max_depth):#max_depth, min_size, depth):
    #print "this is node:   "  +str(type(node))
    left = node['left']
    right = node['right']
    
    if left.shape[0] == 0 and right.shape[0] != 0:
        #print "this is when left.shape[0] == 0 and right.shape[0] != 0 "
        node['left'] = terminalMajorityVote(right[:,-1])
        node['right'] = terminalMajorityVote(right[:,-1])
    if left.shape[0] != 0 and right.shape[0] == 0:
        #print "left.shape[0] != 0 and right.shape[0] == 0 "
        node['left'] = terminalMajorityVote(left[:,-1])
        node['right'] = terminalMajorityVote(left[:,-1])
        return
    ####max depth check
    if depth >= max_depth:
        node['right'], node['left'] = terminalMajorityVote(right[:,-1]), terminalMajorityVote(left[:,-1])
        return
    ##minmum branch check
    if left.shape[0] <= 5:
        node['left'] = terminalMajorityVote(left[:,-1])
        if right.shape[0] <= 5:
            node['right'] = terminalMajorityVote(right[:,-1])
            return
        else:
            node['right'] = getSplit(right)
            treesplit(node['right'],depth + 1,5)
    else:
        node['left'] = getSplit(left)
        

        treesplit(node['left'], depth + 1, 5)
        #print "splitleft"
        if right.shape[0] <= 5:
            node['right'] = terminalMajorityVote(right[:,-1])
            return
        else:
            ##for threshold label right
#             Label = (min(uniqueSet) + max(uniqueSet))/2
#             uniqueSet = [x for x in uniqueSet if x> Label]
            node['right'] = getSplit(right)

            treesplit(node['right'], depth + 1,5)


def build_tree(dataset):
    #unique = [3,4,5,6,7,8,9]
    root = getSplit(dataset)
    #print "root is :" + str(root)
    
    treesplit(root,1,5)
    return root

def predictionOfOne(dataset,node,eachInstance):
    threshold_list = getThreshold2(dataset)[2]
    ##node is the tree
    #print threshold_list
    #print "node['attribute']" + str(node['attribute'])
    #print eachInstance[node['attribute']]
    if eachInstance[node['attribute']] > threshold_list[node['attribute']]:
        if isinstance(node['right'], dict):
            return predictionOfOne(dataset,node['right'], eachInstance)
        else:
            return node['right']
    else:
        
        #print "else" 
        if isinstance(node['left'], dict):
            return predictionOfOne(dataset,node['left'], eachInstance)
        else:
            return node['left']

def accuracy(tree, validset):
    mytree = tree
    predict_y = []
    count = 0
    valid_y = validset[:,-1]
    for row in validset:
        predict = predictionOfOne(validset, mytree, row)
        predict_y.append(predict)
    for i in range(len(predict_y)):
        if predict_y[i] ==  valid_y[i]:
            count +=1
    acc = count / len(valid_y)
    return acc

def f1score(tree, validset):
    valid_y = validset[:,-1].astype(int)
    mytree = tree
    #print 'this is real y:'+ str(valid_y)
    unique = np.unique(valid_y).astype(int)
   # print 'this is unique:   ' + str(unique)
    f1score_list = []
    predict_y = []
    for row in validset:
        predict = predictionOfOne(validset, mytree, row)
        predict_y.append(predict)
    predict_y = np.asarray(predict_y)
    #print 'this is predict:'
    #print predict_y
    for j in unique:
        #print 'current label: '+ str(j)
        TP = np.count_nonzero((valid_y == j) & (predict_y == j))
        #print 'TP'+str(TP)
        FP = np.count_nonzero((valid_y != j) & (predict_y == j))
        #print 'FP'+str(FP)
        FN = np.count_nonzero((valid_y == j) & (predict_y != j))
        #print 'FN'+str(FN)
        f1 = 2*TP/(2*TP+FP+FN)
        #print 'f1'+str(f1)
        f1score_list.append(f1)
    #print 'this is f1score list: '
    #print f1score_list
    return sum(f1score_list)/len(f1score_list)


def main():
    #os.chdir('C:/Users/Henry/Desktop/cs578/hw1')
    my_data = np.genfromtxt('winequality-white.csv', delimiter=';')
    my_data = my_data[1:]
    ##comment the line below to get to the full dataset
    my_data = my_data[np.random.choice(my_data.shape[0], 300, replace=False), :]
    # print my_data
    ##my_data = my_data.tolist()
    label = my_data[:, 11]
    X = my_data[:, 0:11]
    y = label.astype(int)
    ##sigmoid
    X = 1 / (1 + np.exp(-X))
    data = np.hstack((X, y.reshape(len(y), 1)))
    fold1, fold2, fold3, fold4 = crossValidation(data, 4)
    tree_fold1 = build_tree(fold1['train_fold'])
    tree_fold2 = build_tree(fold2['train_fold'])
    tree_fold3 = build_tree(fold3['train_fold'])
    tree_fold4 = build_tree(fold4['train_fold'])

    print 'Hyper-paremeters:'
    print 'Max_Depth: 10'
    print '  '
    ###fold-1
    fold1_train_acc = accuracy(tree_fold1, fold1['train_fold']) * 100
    fold1_train_f1 = f1score(tree_fold1, fold1['train_fold']) * 100
    fold1_valid_acc = accuracy(tree_fold1, fold1['valid_fold']) * 100
    fold1_valid_f1 = f1score(tree_fold1, fold1['valid_fold']) * 100
    fold1_test_acc = accuracy(tree_fold1, fold1['test_fold']) * 100
    fold1_test_f1 = f1score(tree_fold1, fold1['test_fold']) * 100
    print 'Fold-1:'
    print 'Training: F1 score: %s, ' % fold1_train_f1 + 'Accuracy: %s1' % fold1_train_acc
    print 'Validation: F1 score: %s, ' % fold1_valid_f1 + 'Accuracy: %s1' % fold1_valid_acc
    print 'Test: F1 score: %s, ' % fold1_test_f1 + 'Accuracy: %s1' % fold1_test_acc
    print '  '
    ###fold-2
    fold2_train_acc = accuracy(tree_fold2, fold2['train_fold']) * 100
    fold2_train_f1 = f1score(tree_fold2, fold2['train_fold']) * 100
    fold2_valid_acc = accuracy(tree_fold2, fold2['valid_fold']) * 100  ##change the k of knn here
    fold2_valid_f1 = f1score(tree_fold2, fold2['valid_fold']) * 100
    fold2_test_acc = accuracy(tree_fold2, fold2['test_fold']) * 100
    fold2_test_f1 = f1score(tree_fold2, fold2['test_fold']) * 100
    print 'Fold-2:'
    print 'Training: F1 score: %s, ' % fold2_train_f1 + 'Accuracy: %s1' % fold2_train_acc
    print 'Validation: F1 score: %s, ' % fold2_valid_f1 + 'Accuracy: %s1' % fold2_valid_acc
    print 'Test: F1 score: %s, ' % fold2_test_f1 + 'Accuracy: %s1' % fold2_test_acc
    print '  '
    ###fold-3
    fold3_train_acc = accuracy(tree_fold3, fold3['train_fold']) * 100
    fold3_train_f1 = f1score(tree_fold3, fold3['train_fold']) * 100
    fold3_valid_acc = accuracy(tree_fold3, fold3['valid_fold']) * 100  ##change the k of knn here
    fold3_valid_f1 = f1score(tree_fold3, fold3['valid_fold']) * 100
    fold3_test_acc = accuracy(tree_fold3, fold3['test_fold']) * 100
    fold3_test_f1 = f1score(tree_fold3, fold3['test_fold']) * 100
    print 'Fold-3:'
    print 'Training: F1 score: %s, ' % fold3_train_f1 + 'Accuracy: %s1' % fold3_train_acc
    print 'Validation: F1 score: %s, ' % fold3_valid_f1 + 'Accuracy: %s1' % fold3_valid_acc
    print 'Test: F1 score: %s, ' % fold3_test_f1 + 'Accuracy: %s1' % fold3_test_acc
    print '  '
    ###fold-4
    fold4_train_acc = accuracy(tree_fold4, fold4['train_fold']) * 100
    fold4_train_f1 = f1score(tree_fold4, fold4['train_fold']) * 100
    fold4_valid_acc = accuracy(tree_fold4,fold4['valid_fold']) * 100  ##change the k of knn here
    fold4_valid_f1 = f1score(tree_fold4, fold4['valid_fold']) * 100
    fold4_test_acc = accuracy(tree_fold4, fold4['test_fold']) * 100
    fold4_test_f1 = f1score( tree_fold4, fold4['test_fold']) * 100
    print 'Fold-4:'
    print 'Training: F1 score: %s, ' % fold4_train_f1 + 'Accuracy: %s1' % fold4_train_acc
    print 'Validation: F1 score: %s, ' % fold4_valid_f1 + 'Accuracy: %s1' % fold4_valid_acc
    print 'Test: F1 score: %s, ' % fold4_test_f1 + 'Accuracy: %s1' % fold4_test_acc
    print '  '
    ###Average
    avg_train_acc = (fold1_train_acc + fold2_train_acc + fold3_train_acc + fold4_train_acc) / 4
    avg_train_f1 = (fold1_train_f1 + fold2_train_f1 + fold3_train_f1 + fold4_train_f1) / 4
    avg_valid_acc = (fold1_valid_acc + fold2_valid_acc + fold3_valid_acc + fold4_valid_acc) / 4
    avg_valid_f1 = (fold1_valid_f1 + fold2_valid_f1 + fold3_valid_f1 + fold4_valid_f1) / 4
    avg_test_acc = (fold1_test_acc + fold2_test_acc + fold3_test_acc + fold4_test_acc) / 4
    avg_test_f1 = (fold1_test_f1 + fold2_test_f1 + fold3_test_f1 + fold4_test_f1) / 4
    print 'Average:'
    print 'Training: F1 score: %s, ' % avg_train_acc + 'Accuracy: %s1' % avg_train_f1
    print 'Validation: F1 score: %s, ' % avg_valid_f1 + 'Accuracy: %s1' % avg_valid_acc
    print 'Test: F1 score: %s, ' % avg_test_f1 + 'Accuracy: %s1' % avg_test_acc

main()

