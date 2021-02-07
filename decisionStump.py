# -*- coding: utf-8 -*-
"""
HW 1: Decision Stump

@author: Yu Zhong
"""
import csv
import collections
import sys

#read the tsv filer and extract the needed columns
def readtsv(file, i):
    inputfile = []
    results = []

    with open(file) as f:
        
        #remove header
        f.readline()
        r = csv.reader(f, delimiter="\t")
  
        for row in r:
            inputfile.append(row[i])
            results.append(row[-1])
    return inputfile, results


def majority_vote(train, results):
    # this step is to seperate group and then work on majority vote
    result1 =  []
    result2 =  []
    for x, val in enumerate(train):
              
        if val == list(set(train))[0]:
           # train1.append(val)
            result1.append(results[x])
        else:
           # train2.append(val)
            result2.append(results[x])
    
    #count the number of each category 
    counts1 = collections.Counter(result1)
    counts2 = collections.Counter(result2)

    # use sorted to extract the value with more votes
    decision1 = sorted(counts1.items(),key = lambda x:x[1],reverse=True)[0][0]
    decision2 = sorted(counts2.items(),key = lambda x:x[1],reverse = True)[0][0]
    
    return decision1, decision2
    
def predictTest(inputfile, train, decision1, decision2):
    prediction = []
    
    for i in inputfile:
        
        if i == list(set(train))[0]:
            prediction.append(decision1)
        else:
            prediction.append(decision2)
            
    return prediction
    
    
def error_rate(prediction, results):
    
    a = 0
    for i in range(len(prediction)):
        if prediction[i] != results[i]:
            a += 1
   
    return  a/ len(prediction)
    
    


if __name__ == "__main__":
    # Read the file names and split index from command line
    
    train = sys.argv[1]
    test  = sys.argv[2]
    index = int(sys.argv[3])
    label_train = sys.argv[4]
    label_test = sys.argv[5]
    metrics = sys.argv[6]
    
    train, train_results = readtsv(train, index)
    decision1, decision2 = majority_vote(train, train_results)
    test, test_results =  readtsv(test, index)
    
    prediction_train = predictTest(train, train, decision1, decision2)
    prediction_test = predictTest(test, train, decision1, decision2)
    
    error_train = error_rate(prediction_train, train_results)
    error_test = error_rate(prediction_test, test_results)
    
    #write the label train
    with open(label_train, 'w') as f:
        f.writelines("%s\n" % line for line in prediction_train)
    
    #write the label test
    with open(label_test, 'w') as f:
        f.writelines("%s\n" % line for line in prediction_test)
    
    #write metrics
   # word = 'error(train): {} \nerror(test): {}'.format(error_train, error_test)
    with open(metrics,'w') as f:
        f.write('error(train): ' + str(error_train) + '\n')
        f.write('error(test): ' + str(error_test))





