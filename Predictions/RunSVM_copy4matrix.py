# -*- coding: utf-8 -*-

import argparse
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score
import ast
from Predictions.CorpusLoder import *
from Predictions.GetTargets import *
from Predictions.BuildDenseRep import *
from Predictions.BuildSparseRep import *
from Predictions.DenoiseChenEtAl import *


"""
Created on Tue Nov 21 17:01:27 2017

@author: eduardofierro (adapted to run outside de main func)

Run SVMs. 

"""


def listParser(list_string):     
    return ast.literal_eval(list_string)

def buildFeatures(data_dir = 'Data' , min_val = 100, file_type = 'gloves',vocab_size = 10000):
    
    print("Loading Train corpus:")
    train_files = np.loadtxt(data_dir + "/train.txt", dtype="str")
    train_corpus = loadCorpus(train_files, data_dir, min_val)
    
    print("Loading Valid corpus:")
    valid_files = np.loadtxt(data_dir + "/valid.txt", dtype="str")
    valid_corpus = loadCorpus(valid_files, data_dir, min_val)   
        
    if file_type == "gloves":

        subdir = "GloVe"
        glove, index_to_word_map, word_to_index_map = load_gloves(data_dir, subdir)
        
        glove = glove[0:vocab_size,:]
        index_to_word_map = dict([(x, index_to_word_map[x]) for x in range(0,vocab_size)])
        word_to_index_map = dict([(index_to_word_map[index], index) for index in index_to_word_map])

        print("Building Train GloVeToCropus:")
        train_corpus_glove = gloveCorpus(train_corpus, glove, word_to_index_map)   
        print("Building Valid GloVeToCropus:")
        valid_corpus_glove = gloveCorpus(valid_corpus, glove, word_to_index_map) 

        return train_corpus_glove, valid_corpus_glove  

    elif file_type == "sparse":
        subdir = "Sparse"
        vocabulary, index_to_word_map, word_to_index_map = load_sparse(data_dir, subdir)

        vocabulary = vocabulary[0:vocab_size]
        index_to_word_map = dict([(x, index_to_word_map[x]) for x in range(0,vocab_size)])
        word_to_index_map = dict([(index_to_word_map[index], index) for index in index_to_word_map])

        print("Building Train Sparse Matrix:")
        train_sparse_mat = sparseCorpus(train_corpus, vocabulary, word_to_index_map) 
        print("Building Valid Sparse Matrix:")
        valid_sparse_mat = sparseCorpus(valid_corpus, vocabulary, word_to_index_map) 
        
        return train_sparse_mat, valid_sparse_mat  
    
    else:
        raise ValueError("Type not recognized")
    
def buildTargets(word_to_use = "['Educación', 'Campo', 'Sistema Financiero', 'Electoral', 'Derechos Humanos', 'Medio Ambiente', 'Laboral']" , data_dir = 'Data/'):
    
    print("Building targets...")
    train_files = np.loadtxt(data_dir + "train.txt", dtype="str")
    valid_files = np.loadtxt(data_dir + "valid.txt", dtype="str") 
    
    data_main = readTable(data_dir) # Name of table is hard-coded as Main.csv
    train_list_temas = getTemaFromList(train_files, data_main) # The column name is hard coded
    valid_list_temas = getTemaFromList(valid_files, data_main) # The column name is hard coded
    
    topics = listParser(word_to_use)  
    
    train_targets = []
    valid_targets = []
    
    for target in topics:
        
        train_targets.append(genTarget(train_list_temas, target))
        valid_targets.append(genTarget(valid_list_temas, target))
    
    # This is a list of topics and 2 lists of numpy arrays. 
    return topics, train_targets, valid_targets
    
def getNonMissingIndexes(train_features, valid_features, file_type = 'gloves'): 
    
    if file_type == "gloves":
        # Here, is a row of NAs
        train_nonmissing_indexes = [x for x in range(0, train_features.shape[0]) if np.isnan(train_features[x][0])==False]
        valid_nonmissing_indexes = [x for x in range(0, valid_features.shape[0]) if np.isnan(valid_features[x][0])==False]

    elif file_type == "sparse":
        # Here, is a row of zeros (no words)
        train_nonmissing_indexes = [x for x in range(0, train_features.shape[0]) if np.sum(train_features[x])>0]
        valid_nonmissing_indexes = [x for x in range(0, valid_features.shape[0]) if np.sum(valid_features[x])>0]    

    else:
        raise ValueError("Type not recognized")    
    
    return(train_nonmissing_indexes, valid_nonmissing_indexes)  
    
def dropMissing(train_features, valid_features, train_targets, valid_targets):
        
    train_indexes, valid_indexes = getNonMissingIndexes(train_features, valid_features)
    train_features = train_features[train_indexes]
    valid_features = valid_features[valid_indexes]
        
    for x in range(0, len(train_targets)):
        train_targets[x] = train_targets[x][train_indexes]
        valid_targets[x] = valid_targets[x][valid_indexes]
    
    return train_features, valid_features, train_targets, valid_targets

def printTargetBalance(topics_, train_targets_): 
    
    print("Target balance on train:")
    for x in range(0, len(topics_)):

        topic = topics_[x]
        total_train = train_targets_[x].shape[0]
        ones_train = np.sum(train_targets_[x])
        percent = round(ones_train/total_train,2)*100
        print("Topic: {}; Percet Ones: {}%".format(topic, percent))

def VocabSizeChecker(vocab_size = 10000 , min_val=50, max_val=10000):

    if vocab_size > max_val: 
        raise ValueError("Vocabulary Size cannot exceed {}".format(max_val))
    if vocab_size < min_val: 
        raise ValueError("Vocabulary Size cannot be lower than {}".format(min_val))  
    
        