# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 00:10:54 2021

@author: zhong
"""
import numpy as np
import csv
import math
import sys

#read the dictionary
def readtsv(file):
    inputfile = []
    
    with open(file) as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
        
            if len(row) == 1:#to read dictionary
                inputfile.append(row[0].split(" "))
            else:# to read regular files
                add = [row[0]]
                add.extend(row[1].split(" "))  
                inputfile.append(add)
            
    return inputfile 

#read the formatted and remove :1
def readtsv2(file):
    inputfile = []
    
    with open(file) as f:
        r = csv.reader(f, delimiter="\t")

        for row in r:
            
            #remove the :1
            for l in range(1,len(row)):
                row[l] = row[l].split(':')[0]
            inputfile.append(row)
               
    return inputfile 

def likelihood(theta, x, y):
    top = (np.exp(np.dot(theta.T, x)))**y
    buttom = 1+np.exp(np.dot(theta.T, x))
    return top/buttom

def objective(theta, x,y ):
    
    part1 = -y*np.dot(theta.T, x) 
    part2 = math.log(1+np.exp(np.dot(theta.T, x)))
    single_obj = part1 + part2
   
    return single_obj

def derivative(x,y, theta):
    parts = np.exp(np.dot(theta.T, x))/(1+ np.exp(np.dot(theta.T, x)))
    return x*(y- parts)

def theta_x(l, dictionary):
    x = np.zeros(len(dictionary) + 1)

    for i in l[1:]:
        x[int(i)] = 1
    return x

def training(theta, inputfile, epoch):
  #  loss = 0
    dev = 0
    i = 0
    """
    for i in range(0,len(inputfile)):
        if n == epoch:
            break
        to_train = inputfile[i]
        x = theta_x(to_train, dictionary)
        y = int(to_train[0])
      #  loss = objective(theta, x, y)
        
        #partial derivative
        dev = derivative(x,y, theta)
        theta += 0.1*dev
        print(theta)
        n +=1
    """
    while i < epoch:
        for k in range(0,len(inputfile)):
         #   print(theta)
            to_train = inputfile[k]
            x = theta_x(to_train, dictionary)
            y = int(to_train[0])
            
            #partial derivative
            dev = derivative(x,y, theta)
         #   print('dev',dev)
            theta += 0.1*dev/len(inputfile)
        
        i +=1
 #   loss = loss/len(inputfile)
#    dev = -dev/len(inputfile)
    return theta

def predict(theta, testfile):
    yhat = []
    for i in testfile:
        x = theta_x(i, dictionary)
        # pred = np.dot(x, theta)
        val = np.dot( theta.T, x)
        
        pred =(1)/(1+ np.exp(-val))
     #   print(pred)
        if pred >0.5:
            yhat.append(1)
        else:
            yhat.append(0)
    return yhat

def error_rate(yhat, true):
    error = 0
    
    for i in range(len(yhat)):
        if yhat[i] != true[i]:
            error += 1
   
    return  error/ len(yhat)
    
 
    

if __name__ == "__main__":
    """
    train = "formatted_train.tsv"
    valid = "formatted_valid.tsv"
    test = "handout/smalloutput/model1_formatted_test.tsv"
    dictfile = "handout/dict.txt"
    label_train = "train_out.labels"
    label_test = "test_out.labels"
    metrics = "metrics_out.txt"
    epoch = 30
    """
    train = sys.argv[1]
    valid = sys.argv[2]
    test = sys.argv[3]
    dictfile = sys.argv[4]
    label_train = sys.argv[5]
    label_test = sys.argv[6]
    metrics = sys.argv[7]
    epoch = int(sys.argv[8])
    
    #dictionary
    dictfile = readtsv(dictfile)
    dictionary = {i[0]:i[1] for i in dictfile}
    
    train = readtsv2(train)
   
    theta = np.zeros(len(dictionary)+1)
    theta = training(theta, train, epoch) 
    yhat_train = predict(theta, train)
    y = np.array([int(i[0]) for i in train])
    error_train = error_rate(yhat_train,y)
    
    
    test = readtsv2(test)
    yhat_test = predict(theta, test)
    y = np.array([int(i[0]) for i in test])
    error_test = error_rate(yhat_test,y)
    
    with open(metrics,'w') as f:
       f.write('error(train): ' + str(error_train) + '\n')
       f.write('error(test): ' + str(error_test))
       
    with open(label_train, 'w') as f:
        f.writelines("%s\n" % line for line in yhat_train)
   
    
    #write the label test
    with open(label_test, 'w') as f:
        f.writelines("%s\n" % line for line in yhat_test)
