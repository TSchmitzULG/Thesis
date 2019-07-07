#!/usr/bin/python
# -*- coding: latin-1 -*-
"""datashaping loading in file at each batch, format [batchSize,numStep]"""
import scipy.io as sio
import numpy as np




def loadValidationGain(matrix,num_step,valSize):
    valInput = matrix[:valSize,0]
    valParameters = matrix[:valSize,2]
    my_indices = np.arange(len(valInput)-(num_step-1)) # all indices
    indices = (np.arange(num_step) +my_indices[:,np.newaxis]) # each sequence of num_step decayed from one sample
    valInput = np.stack((np.take(valInput,indices),np.take(valParameters,indices)),axis=2) #dim [numsample,num_step,num_features]
    valOutput = np.reshape(matrix[num_step-1:valSize,1],(valSize-(num_step-1),1))
    
    return valInput,valOutput

# if memory cannot have all the dataset in it, we feed by minibatch
def loadBatch(matrix,num_step):
    inp = matrix[:,0]
    outp = np.reshape(matrix[num_step-1:,1],(len(inp)-(num_step-1),1))
    matrixParameters = matrix[:,2]
    my_indices = np.arange(len(inp)-(num_step-1))
    indices = (np.arange(num_step) +my_indices[:,np.newaxis])
    #print("length indices {}".format ((indices)))
    #inp = np.take(inp,indices)
    inp = np.stack((np.take(inp,indices),np.take(matrixParameters,indices)),axis=2)
    return inp,outp

def loadInputOutputGain(matrix,num_step,index,subsize):
    subsizeIn = int (np.floor(len(matrix)/subsize)) # if all the dataset does not fit in the memory we have splitted it into 10 subatches

    #for training
    matrixIn = matrix[index*subsizeIn:(index+1)*subsizeIn,0]
    matrixOut = matrix[index*subsizeIn:(index+1)*subsizeIn,1]
    matrixParameters = matrix[index*subsizeIn:(index+1)*subsizeIn,2]
    my_indices = np.arange(len(matrixIn)-(num_step-1)) # all indices - numstep
    indices = (np.arange(num_step) +my_indices[:,np.newaxis]) # each sequence of num_step decayed from one sample
    inp = np.stack((np.take(matrixIn,indices),np.take(matrixParameters,indices)),axis=2) # [numsample,num_step,num_features]
    outp = np.reshape(matrixOut[num_step-1:],(len(matrixIn)-(num_step-1),1))
    
    return inp,outp

def loadInputOutput2(matrix,num_step):
    matrixIn = matrix[:,0]
    matrixOut = matrix[:,1]
    matrixParameters = matrix[:,2]
    reshapedInput  = []
    reshapedOutput = []
    nbExample = len(matrixIn)-num_step
    for i in range(nbExample):
        temp_list = matrixIn[i:i+num_step]
        temp_list_out = [matrixOut[i+num_step-1]]
        reshapedInput.append(np.array(temp_list))
        reshapedOutput.append(np.array(temp_list_out))
    return reshapedInput,reshapedOutput

def loadInputOutputGain2(matrix,num_step):
    #for training
    matrixIn = matrix[:,0]
    matrixOut = matrix[:,1]
    matrixParameters = matrix[:,2]
    my_indices = np.arange(len(matrixIn)-(num_step-1)) # all indices - numstep
    indices = (np.arange(num_step) +my_indices[:,np.newaxis]) # each sequence of num_step decayed from one sample
    inp = np.stack((np.take(matrixIn,indices),np.take(matrixParameters,indices)),axis=2) # [numsample,num_step,num_features]
    outp = np.reshape(matrixOut[num_step-1:],(len(matrixIn)-(num_step-1),1))
    
    return inp,outp

def shuffleMatrix(data):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    return [data[i] for i in shuffled_indices]

def loadInputOutputGainShuffle(matrix,num_step):
    #for training
    matrixIn = matrix[:,0]
    matrixOut = matrix[:,1]
    matrixParameters = matrix[:,2]
    my_indices = np.arange(len(matrixIn)-(num_step-1)) # all indices - numstep
    indices = (np.arange(num_step) +my_indices[:,np.newaxis]) # each sequence of num_step decayed from one sample
    inp = shuffleMatrix(np.stack((np.take(matrixIn,indices),np.take(matrixParameters,indices)),axis=2)) # [numsample,num_step,num_features]
    outp = shuffleMatrix(np.reshape(matrixOut[num_step-1:],(len(matrixIn)-(num_step-1),1)))
    
    return inp,outp