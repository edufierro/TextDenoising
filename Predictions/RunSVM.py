# -*- coding: utf-8 -*-

import argparse
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score
import ast
from CorpusLoder import loadCorpus
from GetTargets import readTable, getTemaFromList, genTarget
from BuildDenseRep import load_gloves, gloveCorpus
from BuildSparseRep import load_sparse, sparseCorpus
from DenoiseChenEtAl import ALM_RoMaCo


"""
Created on Tue Nov 21 17:01:27 2017

@author: eduardofierro

Run SVMs. 

"""

parser = argparse.ArgumentParser()
parser.add_argument('--main_data_dir', type=str, default='/Users/eduardofierro/Google Drive/TercerSemetre/Optimization/Project/Data/', help='Main data dir')
parser.add_argument('--min_value', type=float, default=100, help='Min sentence length to consider')
parser.add_argument('--file_type', type=str, default="gloves", help='gloves := GloVe; sparse=Sparse (normal)')
parser.add_argument('--list_topics', type=str, default="['Educación', 'Campo', 'Sistema Financiero', 'Electoral', 'Derechos Humanos', 'Medio Ambiente', 'Laboral']", help='A list of topics to chose from, as string')
parser.add_argument('--SVM_hyperparam', type=float, default=0.001, help='SVM hyperparam (for all topics)')
parser.add_argument('--SVM_kernel', type=str, default="linear", help="SVM kernel. Default = linear; Must be ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable ")
parser.add_argument('--vocab_size', type=int, default=10000, help="Vocabulary size to use for classification task")
parser.add_argument('--denoise', type=str, default='None', help="Denoising techique to use - Accepted: None; ChetEtAl")
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
        
    if file_type == "gloves":

        subdir = "GloVe"
        glove, index_to_word_map, word_to_index_map = load_gloves(data_dir, subdir)
        
        glove = glove[0:opt.vocab_size,:]
        index_to_word_map = dict([(x, index_to_word_map[x]) for x in range(0,opt.vocab_size)])
        word_to_index_map = dict([(index_to_word_map[index], index) for index in index_to_word_map])

        print("Building Train GloVeToCropus:")
        train_corpus_glove = gloveCorpus(train_corpus, glove, word_to_index_map)   
        print("Building Valid GloVeToCropus:")
        valid_corpus_glove = gloveCorpus(valid_corpus, glove, word_to_index_map) 

        return train_corpus_glove, valid_corpus_glove  

    elif file_type == "sparse":
        subdir = "Sparse"
        vocabulary, index_to_word_map, word_to_index_map = load_sparse(data_dir, subdir)

        vocabulary = vocabulary[0:opt.vocab_size]
        index_to_word_map = dict([(x, index_to_word_map[x]) for x in range(0,opt.vocab_size)])
        word_to_index_map = dict([(index_to_word_map[index], index) for index in index_to_word_map])

        print("Building Train Sparse Matrix:")
        train_sparse_mat = sparseCorpus(train_corpus, vocabulary, word_to_index_map) 
        print("Building Valid Sparse Matrix:")
        valid_sparse_mat = sparseCorpus(valid_corpus, vocabulary, word_to_index_map) 
        
        return train_sparse_mat, valid_sparse_mat  
    
    else:
        raise ValueError("Type not recognized")
    
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
    
def getNonMissingIndexes(train_features, valid_features, file_type = opt.file_type): 
    
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
    for x in range(0, len(topics)):

        topic = topics_[x]
        total_train = train_targets_[x].shape[0]
        ones_train = np.sum(train_targets_[x])
        percent = round(ones_train/total_train,2)*100
        print("Topic: {}; Percet Ones: {}%".format(topic, percent))

def VocabSizeChecker(vocab_size=opt.vocab_size , min_val=50, max_val=10000):

    if vocab_size > max_val: 
        raise ValueError("Vocabulary Size cannot exceed {}".format(max_val))
    if vocab_size < min_val: 
        raise ValueError("Vocabulary Size cannot be lower than {}".format(min_val))  
    
if __name__ == '__main__':

    VocabSizeChecker()
    train_features, valid_features = buildFeatures()
    topics, train_target, valid_target = buildTargets()
    train_features, valid_features, train_target, valid_target = dropMissing(train_features, valid_features, train_target, valid_target)
    printTargetBalance(topics, train_target)

    if opt.denoise == "None": 
        print("No Denoising method selected")
    elif opt.denoise == "ChetEtAl":
        print("Denoising Train using Chen et.Al 2011")
        train_features, train_noise = ALM_RoMaCo(train_features)
        print("Denoising valid using Chen et.Al 2011")
        valid_features, valid_noise = ALM_RoMaCo(valid_features)

    print("\n")
    print("Running SVMs; Type = {}; Regularization = {}; Kernel = {}; Vobab Size = {}; Denoising = {}".format(opt.file_type, opt.SVM_hyperparam, opt.SVM_kernel, opt.vocab_size, opt.denoise))

    for x in range(0, len(topics)): 

        my_svm = svm.SVC(kernel=opt.SVM_kernel, C=opt.SVM_hyperparam, probability=True)
        
        my_svm.fit(train_features, train_target[x])         
        acc_train = my_svm.score(train_features, train_target[x])
        acc_valid = my_svm.score(valid_features, valid_target[x])

        y_proba_train =  my_svm.predict_proba(train_features)
        y_proba_valid =  my_svm.predict_proba(valid_features)
        auc_train = roc_auc_score(train_target[x], y_proba_train[:,1])
        auc_valid = roc_auc_score(valid_target[x], y_proba_valid[:,1])

        print("Topic {} - Acc on train: {}".format(topics[x], acc_train))
        print("Topic {} - Acc on validation: {}".format(topics[x], acc_valid))
        print("Topic {} - AUC on train: {}".format(topics[x], auc_train))
        print("Topic {} - AUC on validation: {}".format(topics[x], auc_valid))

        