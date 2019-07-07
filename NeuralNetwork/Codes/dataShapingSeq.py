#!/usr/bin/python
# -*- coding: latin-1 -*-
"""datashaping loading in file at each batch, format [batchSize,numStep]"""
import scipy.io as sio
import numpy as np



def loadTrainTest(matrixTrain,matrixTest,num_step,num_class,maxSize):
    
    trainInput = matrixTrain[:maxSize,0]
    my_indices = np.arange(len(trainInput)-(num_step-1))
    indices = (np.arange(num_step) +my_indices[:,np.newaxis])
    trainInput = np.take(trainInput,indices)
    
    my_indices = np.arange(len(trainInput)-(num_step-1))
    indices = (np.arange(num_step-num_class,num_step) +my_indices[:,np.newaxis])
    trainOutput = np.take(matrixTrain[:,1],indices)
    
    sizeTest = len(matrixTest)
    if maxSize < sizeTest: sizeTest=maxSize
    testInput = matrixTest[:sizeTest,0]
    my_indices = np.arange(len(testInput)-(num_step-1))
    indices = (np.arange(num_step) +my_indices[:,np.newaxis])
    testInput = np.take(testInput,indices)
    
    my_indices = np.arange(len(testInput)-(num_step-1))
    indices = (np.arange(num_step-num_class,num_step) + my_indices[:,np.newaxis])
    testOutput = np.take(matrixTest[:,1],indices)
    return trainInput,trainOutput,testInput,testOutput

def loadValidation(matrix,num_step,valSize):
    maxSize = len(matrix)
    valInput = matrix[:valSize,0]
    valOutput = matrix[:valSize,1]
    my_indices = np.arange(len(valInput)-(num_step-1))
    indices = (np.arange(num_step) +my_indices[:,np.newaxis])
    valInput = np.take(valInput,indices)
    valOutput = np.reshape(valOutput[num_step-1:valSize],(valSize-(num_step-1),1))
    
    return valInput,valOutput

def loadValidationSeq(matrix,num_step,valSize,num_class):
    
    valInput = matrix[:valSize,0]
    valOutput = matrix[:valSize,1]
    my_indices = np.arange(0,len(valInput)-(num_step-1),num_class)
    indices = (np.arange(num_step) +my_indices[:,np.newaxis])
    #print("indices in {}".format(indices))
    valInput = np.take(valInput,indices)

    my_indices = np.arange(num_step-1,valSize,num_class)
    indices = (my_indices[:,np.newaxis]-np.arange(num_class-1,-1,-1))
    #print("indices out {}".format(indices))
    valOutput = np.take(valOutput,indices)
    
    return valInput,valOutput