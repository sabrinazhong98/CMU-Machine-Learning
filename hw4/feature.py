# -*- coding: utf-8 -*-
"""
hw4 feature.py

@author: Yu Zhong
"""

import csv
import sys
import collections


#read the file
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

def feature(flag, file, dictionary):
    """
    if flag == 1:
        flag1 = []
       
        
        for line in file:
            
            flag_words = str(line[0])
            flag1_sub =[]
            for word in line[1:]:
                if word in dictionary.keys() and dictionary[word] not in flag1_sub:
                    flag_words += "\t" + str(dictionary[word]) + ":1"
                    flag1_sub.append(dictionary[word])
         #   flag_words +="\n"
            flag1.append(flag_words)
        return flag1
    """
    if flag == 1:
        
        flag1 = [] 
        
        for line in file:
            
            flag1_sub =[]
            temporary =[line[0]]
            for word in line[1:]:
                if word in dictionary.keys() and dictionary[word] not in flag1_sub:
                    temporary.append(str(dictionary[word]) + ":1")
                    flag1_sub.append(dictionary[word])
         
            flag1.append(temporary)
        return flag1
    elif flag == 2:
       
        flag2 = []
        
        for line in file:
            flag2_sub = []
            count_words = collections.Counter(line[1:])
            temporary =[line[0]]
            for word in line[1:]:
                if word in dictionary.keys() and count_words[word] < 4 and dictionary[word] not in flag2_sub:
                    
                    temporary.append(str(dictionary[word])+ ":1")
                    flag2_sub.append(dictionary[word])
            flag2.append(temporary)
        return flag2
        """
        for word in line[1:]:
            
            if word in dictionary.keys() and count_words[word] < 4:
                flag_words += "\t" + str(dictionary[word]) + ":1" 
        flag_words +='\n'
        flag2.append(flag_words)
        """
            

def list_to_str(flaglist):
    big_string =''
    for i in flaglist:
        str1 = str(i[0]) + '\t'
        str2 = ('\t').join(i[1:])+'\n'
        big_string += str1 + str2
    return big_string

    
    
if __name__ == "__main__":
    
    # read all the files needed
    """
    train = "handout/smalldata/train_data.tsv"
    valid = "handout/smalldata/valid_data.tsv"
    test = "handout/smalldata/test_data.tsv"
    dictfile = "handout/dict.txt"
    name_train = "formatted_train.tsv"
    name_valid = "formatted_valid.tsv"
    name_test = "formatted_test.tsv"
    model = 2
    """
    train = sys.argv[1]
    valid = sys.argv[2]
    test = sys.argv[3]
    dictfile = sys.argv[4]
    name_train = sys.argv[5]
    name_valid = sys.argv[6]
    name_test = sys.argv[7]
    model = int(sys.argv[8])
    
    #read dictionary
    dictfile = readtsv(dictfile)
    dictionary = {i[0]:i[1] for i in dictfile}
    
    #train file
    train = readtsv(train)
    valid = readtsv(valid)
    test = readtsv(test)
    
    
    #generate file
    flag_train = feature(model, train, dictionary)
    flag_valid = feature(model, valid, dictionary)
    flag_test = feature(model, test, dictionary)
    

    to_write_train = list_to_str(flag_train)
    to_write_valid = list_to_str(flag_valid)
    to_write_test = list_to_str(flag_test)
    
    with open(name_train, 'w') as f:
        f.write(to_write_train)
    
    with open(name_valid, 'w') as f:
        f.write(to_write_valid)
    
    with open(name_test, 'w') as f:
        f.write(to_write_test)
        
    
        
        
        
        
        
        
        
        
