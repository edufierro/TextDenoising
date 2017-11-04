# -*- coding: utf-8 -*-

import os
import numpy as np

"""
Created on Sat Nov  4 13:58:00 2017

@author: eduardofierro

Purpose : Optimization: Train/Valid/Test Split of documents

Generates 3 text files with file names in of test, train and valid
"""

##### File params ##### 

data_dir = "/Users/eduardofierro/Google Drive/TercerSemetre/Optimization/Project/Data"
save_dir = data_dir
train = 70 # Means N%. Ie 70 = 70%
valid = 15 # Means N%. Ie 70 = 70%
numpys_seed = 1234

##### File params ##### 

def getFileNames(datadir): 
    
    '''
    Reads text files names, returns numpy array

    params:
    @datadir = Local dir where the data is located   
    
    returns: 
    @files = numpy array with names of files in directory
    
    '''
    
    files = os.listdir(datadir + "/TXTsOriginal")
    files = np.array(files)
     
    return files

def clean_list(fileArray): 
    
    '''
    Gets a list of files, removes files that start with a dot 
    (i.e. '.DS_Store')
    
    params: 
    @filesArray: Array of files
    
    returns: 
    @filesArray: Array with no file names that start with a dot
    '''
    
    mask = [x[0] != "." for x in fileArray]
    fileArray = fileArray[mask]
    return fileArray
    
    
def id_splitter(fileArray, per_train, per_valid, random_seed=numpys_seed):
    
    '''
    Function to split an id_List into Train, Valid and Test. 
    
    Params: 
    @fileArray - Array of ids
    @per_train - Percentage on train, in format 0 to 100
    @per_valid - Percentage on validation, in format 0 to 100
    @random_seed - Numpys random seed.
    
    Return: 
    @train, @valid, @test - id_List randomly splitted into this 3 sets.     
    '''
    
    if per_train < 0: 
        raise ValueError('per_train out of bound: Selected {} but need a number between 0 and 100'.format(per_train))
     
    if per_train > 100: 
        raise ValueError('per_train out of bound: Selected {} but need a number between 0 and 100'.format(per_train))
        
    if per_valid < 0: 
        raise ValueError('per_valid out of bound: Selected {} but need a number between 0 and 100'.format(per_valid))
        
    if per_valid  > 100: 
        raise ValueError('per_valid out of bound: Selected {} but need a number between 0 and 100'.format(per_valid))
        
    if per_train + per_valid > 100: 
        raise ValueError('per_valid and per_train add more than 100.')
        
    np.random.seed(random_seed)    
    random = np.random.randint(0, 100, fileArray.shape[0])
    
    train = fileArray[random < per_train]
    valid = fileArray[(random >= (per_train)) & (random < (per_train + per_valid))]
    
    if (per_train + per_valid) < 100:
        test = fileArray[random >= (per_train + per_valid)]
        return(train.tolist(), valid.tolist(), test.tolist())
        
    else: 
        return(train.tolist(), valid.tolist(), None)
     
def main(print_result=True):
    files = getFileNames(data_dir)
    files = clean_list(files)
    train_files, valid_files, test_files = id_splitter(files, train, valid)
    
    file = open(save_dir + "/train.txt", 'w')
    for line in train_files: 
        file.write(line + "\n") 
    file.close

    file = open(save_dir + "/valid.txt", 'w')
    for line in valid_files: 
        file.write(line + "\n") 
    file.close

    if  test_files:
        file = open(save_dir + "/test.txt", 'w')
        for line in test_files:
            file.write(line + "\n") 
        file.close
    
    if print_result: 
        print("Train: {} files ({}%)".format(len(train_files), round(len(train_files)*100/files.shape[0], 2)))
        print("Valid: {} files ({}%)".format(len(valid_files), round(len(valid_files)*100/files.shape[0], 2)))
        if test_files: 
            print("Test: {} files ({}%)".format(len(test_files), round(len(test_files)*100/files.shape[0], 2)))
        else: 
            print("No test file generated")
            
if __name__ == '__main__':
    main()