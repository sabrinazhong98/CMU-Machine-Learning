# -*- coding: utf-8 -*-
"""
HW2: decision tree

@author: Yu Zhong
"""
import csv
import sys
import math
import collections

#read the tsv filer and extract the needed columns
def readtsv(file):
    inputfile = []
    results = []

    with open(file) as f:
        
        #remove header
        header = f.readline()
        header = header.split('\t')
        r = csv.reader(f, delimiter="\t")
  
        for row in r:
            inputfile.append(row[0:len(row)-1])
            results.append(row[-1])
    return inputfile, results, header


def countValue(results):
    counts = collections.Counter(results)
       
    return counts

def getEntropy(counts):
    entropy = 0
    for i in counts.values():
        p = i/sum(counts.values())
        addup = p*math.log(p,2)
        entropy -= addup
    return entropy

def findBestSplit(inputfile, results, parent_entropy):
    #number of columns
    ncol = len(inputfile[0])
    
    possible_results = []
    #get possible results
    for a in inputfile:
        for b in a:
            if b not in possible_results:
                possible_results.append(b)

    info_columns = {}
    for j in range(ncol):
        
        if len(possible_results) > 1:
            
            #single column, 2 output
            column1_results = [results[i] for i in range(len(results)) if inputfile[i][j] == possible_results[0]]
            count1 = countValue(column1_results)
            entropy1 = getEntropy(count1)
            column2_resutls = [results[i] for i in range(len(results)) if inputfile[i][j] == possible_results[1]]   
            count2 = countValue(column2_resutls)
            entropy2 = getEntropy(count2)
           
            total = sum(count1.values()) + sum(count2.values())
            
            info_gain = parent_entropy - entropy1 * sum(count1.values())/total - entropy2 * sum(count2.values())/total 
            
        
        elif len(possible_results) == 1:
     
            column1_results = [results[i] for i in range(len(results)) if inputfile[i][j] == possible_results[0]]
            count1 = countValue(column1_results)
      
            entropy1 = getEntropy(count1)
         
            entropy2 = 0
            
            total = sum(count1.values())
            info_gain = parent_entropy - entropy1 * sum(count1.values())/total 
           
    
        info_columns[j] = info_gain
        
        
    return sorted(info_columns.items(),key = lambda x:x[1],reverse=True)[0] 


class TreeNode(object):
    result = []
    def __init__(self, val):
        self.val = val
        self.leftNode = None
        self.rightNode = None

def printTree(root, fullresult,level=0):
    if root:
        if 'h' not in root.val:
            print('[',root.val['c'][fullresult[1]],' ',fullresult[1],'/' ,root.val['c'][fullresult[0]],' ',fullresult[0],']',sep = '')
        else:
            split = root.val['h']
            for f in fullresult:
                if f not in root.val['c']:
                    root.val['c'][f] = 0
            condition1 = str(root.val['c'][fullresult[1]]) + ' '+ fullresult[1]
            condition2 = str(root.val['c'][fullresult[0]]) + ' '+ fullresult[0]
            head = split[0]+ ' = '+ root.val['n']+ ': '
            print("| "*level,head, '[', condition1,'/',condition2, ']',sep = '')
    else:
        return
    printTree(root.leftNode, fullresult, level + 1)
    printTree(root.rightNode, fullresult, level + 1)
    
    
class BinaryTree(object):
 
    def __init__(self,maxdepth):
        
        self.depth = 0
        self.tree = None
        self.maxdepth = maxdepth
      
       
    def buildTree(self,node,split_col,inputfile, result,fullresult, infogain,counts, header,parentEntropy,depth = 0 ):
        

        if len(counts) == 1:
       
            return node
        elif self.maxdepth == 0:
            return node
        elif depth >= self.maxdepth:
            
            return node
       
        elif len(header) == 0:
            
            return node
        elif len(list(set(i[split_col] for i in inputfile))) == 1:
          
            return node
        else:
            
            #header for passing on
            header_pass = [val for h, val in enumerate(header) if h !=split_col]
            
            #headr for printing
            header_print = [val for h, val in enumerate(header) if h ==split_col]
            
            #find the condition 'A' or 'NOT A'    
            node_condition = list(set([i[split_col] for i in inputfile]))
            
            #extract the result with 'A' condition or 'NOT A' condition

            result1 = [result[n] for n in range(len(result)) if inputfile[n][split_col] == node_condition[1]]
            inputfile1 = [k for k in inputfile if k[split_col] == node_condition[1]]
            
                       
            result2= [result[n] for n in range(len(result)) if inputfile[n][split_col] == node_condition[0]]
            inputfile2 = [k for k in inputfile if k[split_col] == node_condition[0]]
            
            
            for n in inputfile1:
                n.pop(split_col)
            
                
            for n in inputfile2:
                n.pop(split_col)
    
            
            counts1 = countValue(result1)
            counts2 = countValue(result2)
            
        
            node1  = TreeNode({'c':counts1, 'n':node_condition[1],'h':header_print})
            node2  = TreeNode({'c':counts2, 'n':node_condition[0],'h':header_print})
            
            
            split_col1 = split_col2 = 0
            infogain1 = infogain2 = 0
            
            #iterate only when it is not empty
            if inputfile1[0] != []:
                split_col1, infogain1 = findBestSplit(inputfile1, result1, parentEntropy)
                
            if inputfile2[0] != []:
                split_col2, infogain2 = findBestSplit(inputfile2,result2, parentEntropy)   
            
            
            #start recursion
            node.leftNode  = self.buildTree(node1, split_col1, inputfile1, result1,fullresult, infogain1, counts1,header_pass, parentEntropy,depth+1)    
            node.rightNode = self.buildTree(node2, split_col2, inputfile2, result2,fullresult, infogain2, counts2,header_pass, parentEntropy,depth+1)
           
            self.depth += 1                  
            return node


def predictTree(root, test, counts):
    #initiate
    if not root.leftNode and not root.rightNode:
        return counts
    else:
        if 'h' not in root.val:
          
            test_column = root.leftNode.val['h'][0]
            index = int([i for i, val in enumerate(header) if val == test_column][0])
    
        else:
            if not root.leftNode and not root.rightNode:
                return counts
        #find the column we are predicting
            test_column = root.leftNode.val['h'][0]
             
            index = int([i for i, val in enumerate(header) if val == test_column][0])
       
        if test[index] == root.rightNode.val['n']:
            counts = root.rightNode.val['c']
            return predictTree(root.rightNode, test, counts)
        else:
    
            counts = root.leftNode.val['c']
            return predictTree(root.leftNode, test, counts)

def error_rate(prediction, results):
    error = 0
    
    for i in range(len(prediction)):
        if prediction[i] != results[i]:
            error += 1
   
    return  error/ len(prediction)
    
 
if __name__ == "__main__":
  
    # Read the file names and split index from command line
    """
    train = sys.argv[1]
    test = sys.argv[2]
    depth = int(sys.argv[3])
    label_train = sys.argv[4]
    label_test = sys.argv[5]
    metrics = sys.argv[6]
    """
    #This is only for experiment
    train = "politicians_train.tsv"
    test = "politicians_test.tsv"
    depth = 3
    label_train = "small_3_train.labels"
    label_test = "small_3_test.labels"
    metrics = "small_3_metrics.txt"
    
    #read all the file and prepare for training
    inputfile, result, header = readtsv(train)
    
    header.pop() #remove the results header
    counts = countValue(result)
    parentEntropy = getEntropy(counts)
    split_col, infogain = findBestSplit(inputfile, result, parentEntropy)
    node_condition = list(set([i[0] for i in inputfile]))
    fullresult = list(set(result))
 
    
    #training
    x = BinaryTree(depth)
    node  = TreeNode({'s':split_col, 'i':infogain, 'c':counts, 'n':node_condition[1]})
    node = x.buildTree(node,split_col,inputfile, result, fullresult,infogain,counts, header,parentEntropy,depth = 0 )
    printTree(node, fullresult)
   
    
    #predicting
    test_file, test_result, test_header = readtsv(test)
    test_preds = []
    
    #predict test
    for i in test_file:
      
        p = predictTree(node, i, node.val['c'])  
        test_preds.append(sorted(p.items(), key = lambda x:x[1],reverse=True)[0][0])
  
    
    train_preds = []
    inputfile, result, header = readtsv(train)
    

    for i in inputfile:

        p = predictTree(node, i, node.val['c'])  
        train_preds.append(sorted(p.items(), key = lambda x:x[1],reverse=True)[0][0])

    
    error_test  = error_rate(test_preds, test_result)
    error_train = error_rate(train_preds, result)
    print('train error', error_train)
    print('test error', error_test)

    #write metrics file
    with open(metrics,'w') as f:
       f.write('error(train): ' + str(error_train) + '\n')
       f.write('error(test): ' + str(error_test))

     #write the label train
    with open(label_train, 'w') as f:
        f.writelines("%s\n" % line for line in train_preds)
   
    
    #write the label test
    with open(label_test, 'w') as f:
        f.writelines("%s\n" % line for line in test_preds)
      
   
    
    

