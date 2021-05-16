# -*- coding: utf-8 -*-
"""
hw5 neural net
@author: Yu Zhong
"""
import csv
import numpy as np
import math
import sys

def readcsv(fname):
    file = []
    label = []
    with open(fname) as f:
        r = csv.reader(f, delimiter=",")
        for row in r:
            file.append(row[1:])
            label.append(row[0])
    return file, label


    
class Sigmoid():
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):

        self.state= 1/(1+np.exp(-x))
        return 1/(1+np.exp(-x))

    def derivative(self):

        s  = self.state
        return s*(1-s)

class Linear():
    def __init__(self, number,in_feature, out_feature):

        if number == 1:
            self.W  = np.random.uniform(-0.1, 0.1, (in_feature, out_feature))
            self.b = np.zeros(out_feature)
        elif number == 2:
            self.W =  np.random.normal(0, 0, (in_feature, out_feature))
            self.b = np.zeros(out_feature)
       
        self.dW = np.zeros((in_feature, out_feature))
        self.db = np.zeros((1, out_feature))
        self.outfeature = out_feature
    
    def forward(self, x):
        
        self.x = x
        self.preds = np.dot(x, self.W) + self.b
        
        return self.preds

    def backward(self, delta):
        transpose_w = self.W.T
        transpose_x = self.x.T
        
        #update weights and bias
    
        self.dW = np.dot(transpose_x, delta)
      #  print('back weights', self.dW, 'x',self.x, 'de', delta)
     
        self.dx = np.dot(delta,transpose_w)
        
        self.db = delta.reshape(-1,1)
        self.db = [i[0] for i in self.db]
        #print('back', self.dW)
        return self.dx
    



class SoftmaxCrossEntropy():

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
 
       # self.logits = np.array([round(n, 2) for n in x[0]])
        self.logits = x
        self.labels = y
      #  np.sum([np.exp(b) for b in val])
    
      #  print('logits', self.logits)
        self.softmax = np.exp(self.logits)/np.sum(np.exp(self.logits))
       # self.softmax = np.around(np.exp(x)/np.sum(np.exp(x), keepdims = True), decimals = 1)
       # print('softmaxbefore', self.logits)
        #print('softmaxafter', self.softmax)
        self.loss = -np.sum(y*np.log(self.softmax))
       # print('softmax', self.loss)
        return self.loss

    def derivative(self):
        
        return self.softmax - self.labels





class MLP(object):
    def __init__(self, in_feature, neurons, lr, init, trainmode = True):
        
      
           self.lyr1  = Linear(init, in_feature, neurons)
           self.activations = Sigmoid()
           self.lyr2 = Linear(init, neurons, 10)
         #  print('lyr', self.lyr2.W)
           
           self.criterion = SoftmaxCrossEntropy()
           self.neurons = neurons
           self.lr = lr
           self.trainmode = True

    def forward(self, x ):
            
           
            val = self.lyr1.forward(x)
            #print('1', val)
            val = self.activations.forward(val)
            #print('2',val)
  
            val = self.lyr2.forward(val)
      
         #   print('3',val)
            self.val = val

            return val
         
    
    def backward(self,labels):
            true_y = [0 ]*10
       
            true_y[labels] = 1
       
            true_y = np.array(true_y).reshape(-1,10)
  
            
            softmax = self.criterion.forward(self.val, true_y)
            
            dy = self.criterion.derivative()
          #  print('1',dy)
            
            dy = self.lyr2.backward(dy)
           # print('2',dy, self.lyr2.db)
            
            dy = dy* self.activations.derivative() 
            #print(dy)
            
            dy = self.lyr1.backward(dy)
 
            
            return softmax
    def preds(self, x ):
        
       
        val = self.lyr1.forward(x)
        #print(val)
        val = self.activations.forward(val)

        val = self.lyr2.forward(val)
        
        val = np.exp(val)/np.sum(np.exp(val), keepdims = True)
       # print('softmax')
  
        
        self.val = val

        return val
        
    def step(self):

        
        self.lyr1.W -= np.multiply(self.lr , self.lyr1.dW)
        self.lyr1.b -= np.multiply(self.lr , self.lyr1.db)
       # print('step',self.lyr1.W, self.lyr1.b)
        
        self.lyr2.W -=  np.multiply(self.lr , self.lyr2.dW)
        self.lyr2.b -= np.multiply(self.lr , self.lyr2.db)
        #print('step',self.lyr2.W, self.lyr2.b)
        
    def zero_grads(self):

        self.lyr1.dW = [0 for i in self.lyr1.dW]
        self.lyr1.db = [0 for i in self.lyr1.db]
        #print('update', self.lyr1.dW)
        
        self.lyr2.dW = [0 for i in self.lyr2.dW]
        #print('update', self.lyr2.dW)
        self.lyr2.db = [0 for i in self.lyr2.db]
    

        
def training(train, label,valid,label2, epochs, init, neurons, lr, in_feature):
    network = MLP(in_feature, neurons, lr, init)
    
    tentropy = []
    ventropy = []
    for e in range(epochs):
    
        print('\n epoch',e)
        for i in range(len(train)):
          #  print('\n train',i)
            
            t = train[i].reshape(1,-1)
            l = int(label[i])
            
            network.forward(t)
            network.backward(l)
            network.step()
            network.zero_grads()
        
        #loss for train
        tloss = 0
        for v in range(len(train)):
            x = train[v].reshape(1,-1)
            y = int(label[v])
            output = network.forward(x)
            
            true_y = [0 ]*10
       
            true_y[y] = 1
       
            true_y = np.array(true_y).reshape(-1,10)
        
            tloss += network.criterion.forward(output, true_y)
            
        print(tloss/len(train))
        tentropy.append(tloss/len(train))
        
     
        vloss = 0
        for v in range(len(valid)):
            x = valid[v].reshape(1,-1)
            y = int(label2[v])
            output = network.forward(x)
            
            true_y = [0 ]*10
       
            true_y[y] = 1
       
            true_y = np.array(true_y).reshape(-1,10)
        
            vloss += network.criterion.forward(output, true_y)
        print(vloss/len(valid))
        ventropy.append(vloss/len(valid))
    return network, tentropy, ventropy
        

def prediction(file, label, model):
    model.trainmode = False
    yhat = []
    for i in range(len(file)):
        output =model.preds(file[i])
        pred = np.argmax(output)
        
        yhat.append(pred)
    return yhat

def error_rate(yhat, true):
    error = 0
    
    for i in range(len(yhat)):
        if int(yhat[i]) != int(true[i]):
            error += 1
   
    return  error/ len(yhat)
            
         
if __name__ == "__main__":
    """
    fname = 'handout/smallTrain.csv' 
    fname2 = 'handout/smallValidation.csv'
    label_train = 'smallTrain_out.labels'
    label_val = 'smallValidation_out.labels'
    metrics = 'smallMetrics_out.txt'
    epoch = 2
    neurons = 4
    choice = 2
    lr = 0.1
    """
    fname = sys.argv[1]
    fname2 = sys.argv[2]
    label_train = sys.argv[3]
    label_val = sys.argv[4]
    metrics = sys.argv[5]
    epoch = int(sys.argv[6])
    neurons = int(sys.argv[7])
    choice = int(sys.argv[8])
    lr = float(sys.argv[9])
    
    file, label = readcsv(fname)
    file2, label2 = readcsv(fname2) 
    train = np.asarray(file, dtype = 'float32')
    valid = np.asarray(file2, dtype = 'float32')
    
    in_feature = train[0].shape[0]
   
    model, tloss, vloss = training(train, label, valid, label2, epoch, choice, neurons, lr, in_feature)
    

    preds = prediction(valid, label2, model)
    preds_train = prediction(train, label, model)
    error = error_rate(preds, label2)
    error_train = error_rate(preds_train, label)
    
    epoch_loss = []
    for i in range(len(tloss)):
        line = "epoch={0} crossentroipy(train): {1}".format(i+1,tloss[i])
        line2 = "epoch={0} crossentroipy(validation): {1}".format(i+1,vloss[i])
        epoch_loss.append(line)
        epoch_loss.append(line2)
    line3 = "error(train): " + str(error_train)
    line4 = "error(validation): " + str(error)
    epoch_loss.append(line3)
    epoch_loss.append(line4)
        
    
    with open(label_train, 'w') as f:
        f.writelines("%s\n" % line for line in preds_train)
   
    
    with open(label_val, 'w') as f:
        f.writelines("%s\n" % line for line in preds)
        
    with open(metrics,'w') as f:
        f.writelines("%s\n" % line for line in epoch_loss)
     
   

   
    
