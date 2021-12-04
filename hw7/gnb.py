# -*- coding: utf-8 -*-
"""
hw 7
test!!

Yu Zhong
"""
import csv
import numpy as np
import sys
#read the data
def readcsv(fname):
    file = []
    label = []
    with open(fname) as f:
   
        r = csv.reader(f, delimiter=",")
        for row in r:
        
            file.append(row[0:len(row)-1])
            label.append(row[len(row)-1])
            
        file = file[1:]
        label = label[1:]
      
        for f in range(len(file)):
            for num in range(len(file[f])):
                file[f][num] = float(file[f][num])
         
    return file, label



def compute_mean(data, label, classes): 
    all_mean = []
    for c in classes: 
        sum_x = []
        for i in range(len(label)):
           # print(i)
            if label[i] == c:
               # print(data[i])
               
                sum_x.append(data[i])
               
        all_mean.append(np.mean(sum_x, axis = 0))
   
    return all_mean
    
def compute_sdsqr(mean, data, label, classes):
    all_sdsqr = []
    for c in range(len(classes)):
        miu  = mean[c]
        class_test = classes[c]

        sum_x = [(data[i]-miu)**2 for i in range(len(label)) if label[i] == class_test]
        sdsqr = np.mean(sum_x, axis = 0)
        
        all_sdsqr.append(sdsqr)
    return all_sdsqr


def probs(mean, sdsqr, new_data, features): 
    new_data = [np.array(n)[features] for n in new_data]
    
    
    all_probs = []
    for i in range(2):
        
        comp1 = 1/np.sqrt(2*np.pi* (sdsqr[i][features]) )
        
        nominator = -(new_data-mean[i][features])**2
        denominator = 2*sdsqr[i][features]
        comp2 = np.exp(nominator/denominator)
        
        all_probs.append(comp1 * comp2)
    
    return all_probs

def feature_select(k, mean):
    diff = {i: abs(mean[0][i] - mean[1][i]) for i in range(len(mean[0])) }
    diff_sort = sorted(diff.items(), key = lambda x: x[1], reverse = True)
    
    
    features = list(dict(diff_sort[:k]).keys())
    diffs = list(dict(diff_sort[:k]).values())
    
    return features, diffs

def log_space(label, classes, p_xy):
    log_py = []
    
    logsum = []
    for c in range(len(classes)):
  
        cl = classes[c]
        py = len([i for i in label if i == cl])/ len(label)
        log_py = np.log(py)
        
        log_pxpy = np.sum(np.log(p_xy[c]), axis = 1)
       
        logsum.append(log_py + log_pxpy)
    
    return logsum

def inference(logsum):
    result = []
    for l in range(len(logsum[0])):
        if logsum[0][l] > logsum[1][l]:
    
            result.append('tool')
        else:
    
            result.append('building')
    return result

def error_rate(yhat, true):
    error = 0
    
    for i in range(len(yhat)):
        if yhat[i] != true[i]:
            error += 1
   
    return  error/ len(yhat)
    
    
if __name__ == "__main__":
    """
    train_input = 'smalldata/small_train.csv'
    test_input = 'smalldata/small_test.csv'
    train_out = 'small_train_out.labels'
    test_out = 'small_test_out.labels'
    metrics_out = 'metrics_out.txt'
    num_voxels = 21764
    """
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_voxels = int(sys.argv[6])
    
    classes = ['tool', 'building']
    
    train, trainlabel = readcsv(train_input)
    test, testlabel = readcsv(test_input)
    
    mean = compute_mean(train, trainlabel, classes)
    sdsqr = compute_sdsqr(mean, train, trainlabel, classes)
    features, diff = feature_select(num_voxels, mean)
    
    
    p_xy = probs(mean, sdsqr, train, features)

    logspace = log_space(trainlabel, classes, p_xy)
    
    result_train = inference(logspace)
    
    error_train = error_rate(result_train, trainlabel)
    
    
    # for test
    p_xy = probs(mean, sdsqr, test, features)

    logspace = log_space(trainlabel, classes, p_xy)
    
    result_test = inference(logspace)
    
    error_test = error_rate(result_test, testlabel)
    
    
    #write file
    with open(metrics_out,'w') as f:
       f.write('error(train): ' + str(error_train) + '\n')
       f.write('error(test): ' + str(error_test))
       
    with open(train_out, 'w') as f:
        f.writelines("%s\n" % line for line in result_train)
   
    
    #write the label test
    with open(test_out, 'w') as f:
        f.writelines("%s\n" % line for line in result_test)
    


    
    

