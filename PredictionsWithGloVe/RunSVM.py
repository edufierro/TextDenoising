# -*- coding: utf-8 -*-

import argparse
import numpy as np
from sklearn import svm
import ast
from CorpusLoder import loadCorpus
from BuildDenseRep import load_gloves, gloveCorpus
from GetTargets import readTable, getTemaFromList, genTarget


"""
Created on Tue Nov 21 17:01:27 2017

@author: eduardofierro

Baseline SVM. 

"""

parser = argparse.ArgumentParser()
parser.add_argument('--main_data_dir', type=str, default='/Users/eduardofierro/Google Drive/TercerSemetre/Optimization/Project/Data/', help='Main data dir')
parser.add_argument('--min_value', type=float, default=100, help='Min sentence length to consider')
parser.add_argument('--file_type', type=str, default="baseline", help='baseline = GloVe')
parser.add_argument('--list_topics', type=str, default="['Educación', 'Campo', 'Laboral']", help='A list of topics to chose from, as string')
parser.add_argument('--SVM_hyperparam', type=float, default=0.001, help='SVM hyperparam (for all topics)')
parser.add_argument('--SVM_kernel', type=str, default="linear", help="SVM kernel. Default = linear; Must be ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable ")
opt = parser.parse_args()
print(opt)

def listParser(list_string):     
    return ast.literal_eval(list_string)

def buildFeatures(data_dir = opt.main_data_dir, min_val = opt.min_value, file_type = opt.file_type):
    
    print("Loading Train corpus:")
    train_files = np.loadtxt(data_dir + "/train.txt", dtype="str")
    train_corpus = loadCorpus(train_files, data_dir, min_val)
    
    print("Loading Valid corpus:")
    valid_files = np.loadtxt(data_dir + "/valid.txt", dtype="str")
    valid_corpus = loadCorpus(valid_files, data_dir, min_val)   
        
    if file_type == "baseline":
        subdir = "GloVe"
    
    else:
        raise ValueError("Type not recognized")
    
    glove, index_to_word_map, word_to_index_map = load_gloves(data_dir, subdir)

    print("Building Train GloVeToCropus:")
    train_corpus_glove = gloveCorpus(train_corpus, glove, word_to_index_map)   
    print("Building Valid GloVeToCropus:")
    valid_corpus_glove = gloveCorpus(valid_corpus, glove, word_to_index_map)  
        
    return train_corpus_glove, valid_corpus_glove  
    
def buildTargets(word_to_use = opt.list_topics, data_dir = opt.main_data_dir):
    
    print("Building targets...")
    train_files = np.loadtxt(data_dir + "/train.txt", dtype="str")
    valid_files = np.loadtxt(data_dir + "/valid.txt", dtype="str") 
    
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
    
def getNonMissingIndexes(train_gloves, valid_gloves): 
    
    train_nonmissing_indexes = [x for x in range(0, train_gloves.shape[0]) if np.isnan(train_gloves[x][0])==False]
    valid_nonmissing_indexes = [x for x in range(0, valid_gloves.shape[0]) if np.isnan(valid_gloves[x][0])==False]
    
    return(train_nonmissing_indexes, valid_nonmissing_indexes)  
    
def dropMissing(train_gloves, valid_gloves, train_targets, valid_targets):
        
    train_indexes, valid_indexes = getNonMissingIndexes(train_gloves, valid_gloves)
    train_gloves = train_gloves[train_indexes]
    valid_gloves = valid_gloves[valid_indexes]
        
    for x in range(0, len(train_targets)):
        train_targets[x] = train_targets[x][train_indexes]
        valid_targets[x] = valid_targets[x][valid_indexes]
    
    return train_gloves, valid_gloves, train_targets, valid_targets
    
if __name__ == '__main__':
    
    train_corpus_glove, valid_corpus_glove = buildFeatures()
    topics, train_target, valid_target = buildTargets()
    train_corpus_glove, valid_corpus_glove, train_target, valid_target = dropMissing(train_corpus_glove, valid_corpus_glove, train_target, valid_target)
    
    print("\n")
    print("Running SVMs; Regularization = {}; Kernel = {}".format(opt.SVM_hyperparam, opt.SVM_kernel))
    
    acc_train = []
    acc_valid = []
    
    for x in range(0, len(topics)): 
        my_svm = svm.SVC(kernel=opt.SVM_kernel, C=opt.SVM_hyperparam)
        
        my_svm.fit(train_corpus_glove, train_target[x])         
        acc_train.append(my_svm.score(train_corpus_glove, train_target[x]))
        acc_valid.append(my_svm.score(valid_corpus_glove, valid_target[x]))

    for x in range(0, len(topics)): 
        print("Topic {} - Acc on train: {}".format(topics[x], acc_train[x]))
        print("Topic {} - Acc on validation: {}".format(topics[x], acc_valid[x]))
        
#Topic Educación - Acc on train: 0.8938533695383856
#Topic Educación - Acc on validation: 0.8814139110604333
#Topic Campo - Acc on train: 0.9627252530239447
#Topic Campo - Acc on validation: 0.9623717217787914
#Topic Laboral - Acc on train: 0.9212540113552209
#Topic Laboral - Acc on validation: 0.9156214367160775